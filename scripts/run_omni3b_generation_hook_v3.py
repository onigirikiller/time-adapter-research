from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from qwen_omni_utils import process_mm_info
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/omni3b_sequential_v2"
V2_ARTIFACT_DIR = ROOT / "artifacts/omni3b_sequential_v2"
HIDDEN_DIR = V2_ARTIFACT_DIR / "hidden_cache"
OUT_DIR = ROOT / "artifacts/omni3b_generation_hook_v3"
PLOT_DIR = ROOT / "output/figures/omni3b_generation_hook_v3"
CACHE_DIR = ROOT / ".cache/huggingface"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
CONDITIONS = [
    "no_time",
    "zero_vector",
    "correct_time_adapter",
    "shuffled_time_adapter",
    "random_norm_matched",
    "non_time_numeric",
    "oracle_explicit_delta",
]
SEED = 20260623
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1 = import_module(ROOT / "scripts/run_omni_sequential_time_adapter.py", "omni_v1_for_hook")
v2 = import_module(ROOT / "scripts/run_omni3b_v2_experiment.py", "omni_v2_for_hook")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fkey(row, condition, layer, alpha, position):
    return f"{row['id']}|{condition}|L{layer}|a{alpha}|{position}"


def load_split_rows(split="test", stage="extra", max_contexts=0):
    rows = [row for row in read_jsonl(DATA_DIR / f"{split}.jsonl") if stage in row["stages"]]
    if max_contexts:
        contexts = []
        seen = set()
        for row in rows:
            if row["context_id"] not in seen:
                seen.add(row["context_id"])
                contexts.append(row["context_id"])
            if len(contexts) >= max_contexts:
                break
        keep = set(contexts)
        rows = [row for row in rows if row["context_id"] in keep]
    return rows


def feature_matrix(rows):
    return v2.feature_matrix(rows, "multi")


def load_stage_arrays(stage="extra", layer=8):
    splits = {split: read_jsonl(DATA_DIR / f"{split}.jsonl") for split in ["train", "validation", "test"]}
    idx = {split: v2.stage_indices(splits[split], stage) for split in splits}
    rows = {split: [splits[split][int(i)] for i in idx[split]] for split in splits}
    hidden = {
        mode: {
            split: np.load(HIDDEN_DIR / f"{mode}_{split}.npy")[idx[split], layer, :]
            for split in splits
        }
        for mode in ["no_time", "explicit"]
    }
    return rows, hidden


def ensure_model_artifacts(stage="extra", layer=8):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"adapter_proxy_stage-{stage}_layer-{layer}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)

    set_seed()
    rows, hidden = load_stage_arrays(stage, layer)
    deltas = {split: hidden["explicit"][split] - hidden["no_time"][split] for split in rows}
    adapter, adapter_summary = v1.train_adapter(
        "generation_hook_multi_feature",
        feature_matrix(rows["train"]),
        deltas["train"],
        feature_matrix(rows["validation"]),
        deltas["validation"],
        feature_matrix(rows["test"]),
        deltas["test"],
        epochs=260,
    )
    correct_delta = {split: v2.adapter_predict(adapter, feature_matrix(rows[split])) for split in rows}
    zero_delta = {split: np.zeros_like(deltas[split]) for split in rows}
    score_probe = np.load(HIDDEN_DIR / "numeric_probe_score.npy")[:, layer, :]
    probe_rows = rows["train"][: len(score_probe)]
    direction = np.mean(score_probe - hidden["no_time"]["train"][: len(probe_rows)], axis=0)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    non_time_delta = {
        split: direction.reshape(1, -1) * np.linalg.norm(correct_delta[split], axis=1, keepdims=True)
        for split in rows
    }
    rng = np.random.default_rng(SEED + 313)
    random_delta = {
        split: v2.random_norm_delta(correct_delta[split])
        for split in rows
    }
    shuffled_delta = {
        split: v2.shuffle_delta(correct_delta[split], rows[split])
        for split in rows
    }
    eval_features = {
        "correct_time_adapter": {split: v2.decision_features(hidden["no_time"][split], correct_delta[split]) for split in rows},
        "shuffled_time_adapter": {split: v2.decision_features(hidden["no_time"][split], shuffled_delta[split]) for split in rows},
        "random_norm_matched": {split: v2.decision_features(hidden["no_time"][split], random_delta[split]) for split in rows},
        "non_time_numeric": {split: v2.decision_features(hidden["no_time"][split], non_time_delta[split]) for split in rows},
        "zero_vector": {split: v2.decision_features(hidden["no_time"][split], zero_delta[split]) for split in rows},
        "oracle_explicit_delta": {split: v2.decision_features(hidden["no_time"][split], deltas[split]) for split in rows},
    }
    train_x = np.vstack([eval_features["correct_time_adapter"]["train"], eval_features["oracle_explicit_delta"]["train"]])
    train_y = np.concatenate([v2.y_labels(rows["train"]), v2.y_labels(rows["train"])])
    proxy_head, proxy_mean, proxy_std, proxy_summary = v2.train_head(
        "generation_hook_proxy_adapter_head",
        train_x,
        train_y,
        eval_features,
        rows,
        epochs=170,
    )
    no_time_features = {"no_time_hidden": {split: hidden["no_time"][split] for split in rows}}
    context_head, context_mean, context_std, context_summary = v2.train_head(
        "generation_hook_proxy_no_time_head",
        hidden["no_time"]["train"],
        v2.y_labels(rows["train"]),
        no_time_features,
        rows,
        epochs=170,
        lr=6e-4,
    )
    artifact = {
        "stage": stage,
        "layer": layer,
        "adapter_state": adapter.state_dict(),
        "proxy_head_state": proxy_head.state_dict(),
        "proxy_mean": torch.as_tensor(proxy_mean),
        "proxy_std": torch.as_tensor(proxy_std),
        "context_head_state": context_head.state_dict(),
        "context_mean": torch.as_tensor(context_mean),
        "context_std": torch.as_tensor(context_std),
        "non_time_direction_score": torch.as_tensor(direction, dtype=torch.float32),
    }
    torch.save(artifact, path)
    return artifact


def build_vectors(rows, stage="extra", layer=8):
    stage_rows, hidden = load_stage_arrays(stage, layer)
    row_index = {row["id"]: i for i, row in enumerate(stage_rows["test"])}
    selected = np.array([row_index[row["id"]] for row in rows], dtype=np.int64)
    artifact = ensure_model_artifacts(stage, layer)
    adapter = v1.FeatureAdapter(5, hidden["no_time"]["test"].shape[1])
    adapter.load_state_dict(artifact["adapter_state"])
    adapter.eval()
    no_time = hidden["no_time"]["test"][selected]
    explicit = hidden["explicit"]["test"][selected]
    oracle = explicit - no_time
    correct = v2.adapter_predict(adapter, feature_matrix(rows))
    zero = np.zeros_like(correct)
    shuffled = v2.shuffle_delta(correct, rows)
    random_delta = v2.random_norm_delta(correct)
    direction = np.asarray(artifact["non_time_direction_score"], dtype=np.float32)
    non_time = direction.reshape(1, -1) * np.linalg.norm(correct, axis=1, keepdims=True)
    return {
        "no_time": zero,
        "zero_vector": zero,
        "correct_time_adapter": correct,
        "shuffled_time_adapter": shuffled,
        "random_norm_matched": random_delta,
        "non_time_numeric": non_time.astype(np.float32),
        "oracle_explicit_delta": oracle.astype(np.float32),
    }, no_time, artifact


def proxy_predictions(rows, vectors, no_time_hidden, artifact):
    proxy_head = v2.Head(no_time_hidden.shape[1] * 2)
    proxy_head.load_state_dict(artifact["proxy_head_state"])
    proxy_head.eval()
    context_head = v2.Head(no_time_hidden.shape[1])
    context_head.load_state_dict(artifact["context_head_state"])
    context_head.eval()
    out = {}
    for condition, delta in vectors.items():
        if condition == "no_time":
            pred, probs = v2.predict(
                context_head,
                np.asarray(artifact["context_mean"], dtype=np.float32),
                np.asarray(artifact["context_std"], dtype=np.float32),
                no_time_hidden,
            )
        else:
            feats = v2.decision_features(no_time_hidden, delta)
            pred, probs = v2.predict(
                proxy_head,
                np.asarray(artifact["proxy_mean"], dtype=np.float32),
                np.asarray(artifact["proxy_std"], dtype=np.float32),
                feats,
            )
        out[condition] = {
            "pred": pred,
            "probs": probs,
        }
    return out


class InjectionHook:
    def __init__(self, vector, alpha=1.0, position="last_token", mode="add"):
        self.vector = vector
        self.alpha = float(alpha)
        self.position = position
        self.mode = mode
        self.calls = 0
        self.stats = []

    def __call__(self, module, inputs, output):
        self.calls += 1
        hidden = output[0] if isinstance(output, tuple) else output
        vec = torch.as_tensor(self.vector, device=hidden.device, dtype=hidden.dtype).view(1, 1, -1) * self.alpha
        target = hidden[:, -1:, :]
        before = target.detach().float()
        vecf = vec.detach().float()
        denom = torch.linalg.norm(before.reshape(1, -1)) * torch.linalg.norm(vecf.reshape(1, -1))
        cosine = torch.sum(before.reshape(1, -1) * vecf.reshape(1, -1)) / torch.clamp(denom, min=1e-12)
        self.stats.append({
            "call": self.calls,
            "hidden_norm": float(torch.linalg.norm(before).cpu()),
            "injected_norm": float(torch.linalg.norm(vecf).cpu()),
            "hidden_injected_cosine": float(cosine.cpu()),
            "seq_len": int(hidden.shape[1]),
        })
        if self.mode == "capture":
            return output
        if self.position == "all_tokens":
            modified = hidden + vec
        else:
            modified = hidden.clone()
            modified[:, -1:, :] = modified[:, -1:, :] + vec
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


def label_prompt(row, prompt_style="minimal"):
    if prompt_style == "defined":
        return (
            "Task: choose the listener timing label for a streaming dialogue system.\n"
            "Use the unfinished utterance, the audio context, and the hidden timer signal.\n"
            "Labels:\n"
            "- WAIT: stay silent because the user is still speaking, repairing, or asked for time.\n"
            "- BACKCHANNEL: give only a short acknowledgement without taking the turn.\n"
            "- SUPPORT: take the turn with a brief supportive or answering response.\n"
            f"User fragment: \"{row['fragment']}\"\n"
            "Answer with exactly one label: WAIT, BACKCHANNEL, or SUPPORT.\n"
            "Label:"
        )
    return (
        "Task: choose the listener timing label for a streaming dialogue system.\n"
        "Use the unfinished utterance, the audio context, and the hidden timer signal.\n"
        "Labels: WAIT / BACKCHANNEL / SUPPORT\n"
        f"User fragment: \"{row['fragment']}\"\n"
        "Answer with exactly one label.\n"
        "Label:"
    )


def build_inputs(processor, row, prompt_style="minimal"):
    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(Path(row["audio_path"]).resolve())},
                {"type": "text", "text": label_prompt(row, prompt_style=prompt_style)},
            ],
        },
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    return processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)


def append_and_move(inputs, append_ids, device, dtype):
    out = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        tensor = value
        if key == "input_ids" and append_ids:
            extra = torch.tensor([append_ids], dtype=tensor.dtype)
            tensor = torch.cat([tensor, extra], dim=1)
        elif key == "attention_mask" and append_ids:
            extra = torch.ones((tensor.shape[0], len(append_ids)), dtype=tensor.dtype)
            tensor = torch.cat([tensor, extra], dim=1)
        if tensor.is_floating_point():
            out[key] = tensor.to(device=device, dtype=dtype)
        else:
            out[key] = tensor.to(device=device)
    return out


def forward_next_logits(model, inputs, append_ids, vector, layer, alpha, position, mode, device, dtype):
    hook = InjectionHook(vector, alpha=alpha, position=position, mode=mode)
    module = model.thinker.model.layers[layer]
    handle = module.register_forward_hook(hook)
    try:
        moved = append_and_move(inputs, append_ids, device, dtype)
        with torch.inference_mode():
            outputs = model.thinker(**moved, use_audio_in_video=False)
        logits = outputs.logits[0, -1, :].detach().float().cpu()
    finally:
        handle.remove()
    stat = hook.stats[0] if hook.stats else {"hidden_norm": 0.0, "injected_norm": 0.0, "hidden_injected_cosine": 0.0, "seq_len": 0}
    stat["hook_calls"] = hook.calls
    return logits, stat


def log_softmax_np(logits):
    arr = logits.numpy().astype(np.float64)
    m = np.max(arr)
    logsum = m + np.log(np.sum(np.exp(arr - m)))
    return arr - logsum


def make_label_candidates(tokenizer, label_surface):
    variants = {
        "plain": lambda label: [label],
        "leading_space": lambda label: [" " + label],
        "newline": lambda label: ["\n" + label],
        "best_plain_space": lambda label: [label, " " + label],
        "best_plain_space_newline": lambda label: [label, " " + label, "\n" + label],
    }[label_surface]
    candidates = {}
    for label in LABELS:
        cands = []
        for surface in variants(label):
            ids = tokenizer(surface, add_special_tokens=False).input_ids
            if ids:
                cands.append({"surface": surface, "ids": ids})
        candidates[label] = cands
    return candidates


def score_labels(model, inputs, tokenizer, label_ids, vector, layer, alpha, position, condition, device, dtype):
    mode = "capture" if condition == "no_time" else "add"
    base_logits, base_stat = forward_next_logits(model, inputs, [], vector, layer, alpha, position, mode, device, dtype)
    base_lp = log_softmax_np(base_logits)
    raw = {}
    avg = {}
    first_token_lp = {}
    chosen_surface = {}
    for label, candidates in label_ids.items():
        best = None
        for candidate in candidates:
            ids = candidate["ids"]
            total = float(base_lp[ids[0]])
            prefix = [ids[0]]
            for next_id in ids[1:]:
                logits, _ = forward_next_logits(model, inputs, prefix, vector, layer, alpha, position, mode, device, dtype)
                lp = log_softmax_np(logits)
                total += float(lp[next_id])
                prefix.append(next_id)
            cand = {
                "raw": total,
                "avg": total / len(ids),
                "first": float(base_lp[ids[0]]),
                "surface": candidate["surface"],
            }
            if best is None or cand["avg"] > best["avg"]:
                best = cand
        raw[label] = best["raw"]
        avg[label] = best["avg"]
        first_token_lp[label] = best["first"]
        chosen_surface[label] = best["surface"]
    vals = np.array([avg[label] for label in LABELS], dtype=np.float64)
    vals = vals - np.max(vals)
    probs = np.exp(vals) / np.sum(np.exp(vals))
    raw_vals = np.array([raw[label] for label in LABELS], dtype=np.float64)
    raw_probs = np.exp(raw_vals - np.max(raw_vals))
    raw_probs = raw_probs / np.sum(raw_probs)
    pred = LABELS[int(np.argmax(probs))]
    raw_pred = LABELS[int(np.argmax(raw_probs))]
    return {
        "generated_label": pred,
        "raw_argmax_label": raw_pred,
        "wait_logprob": raw["WAIT"],
        "backchannel_logprob": raw["BACKCHANNEL"],
        "support_logprob": raw["SUPPORT"],
        "wait_avg_logprob": avg["WAIT"],
        "backchannel_avg_logprob": avg["BACKCHANNEL"],
        "support_avg_logprob": avg["SUPPORT"],
        "wait_prob": float(probs[0]),
        "backchannel_prob": float(probs[1]),
        "support_prob": float(probs[2]),
        "wait_raw_prob": float(raw_probs[0]),
        "backchannel_raw_prob": float(raw_probs[1]),
        "support_raw_prob": float(raw_probs[2]),
        "wait_first_token_logprob": first_token_lp["WAIT"],
        "backchannel_first_token_logprob": first_token_lp["BACKCHANNEL"],
        "support_first_token_logprob": first_token_lp["SUPPORT"],
        "wait_surface": chosen_surface["WAIT"],
        "backchannel_surface": chosen_surface["BACKCHANNEL"],
        "support_surface": chosen_surface["SUPPORT"],
        **base_stat,
    }


def completed_keys(path):
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["key"] for row in csv.DictReader(f)}


CSV_FIELDS = [
    "key",
    "row_index",
    "id",
    "context_id",
    "profile",
    "fragment",
    "silence_seconds",
    "gold_label",
    "condition",
    "generated_label",
    "raw_argmax_label",
    "proxy_label",
    "proxy_wait_prob",
    "proxy_backchannel_prob",
    "proxy_support_prob",
    "wait_logprob",
    "backchannel_logprob",
    "support_logprob",
    "wait_avg_logprob",
    "backchannel_avg_logprob",
    "support_avg_logprob",
    "wait_prob",
    "backchannel_prob",
    "support_prob",
    "wait_raw_prob",
    "backchannel_raw_prob",
    "support_raw_prob",
    "wait_first_token_logprob",
    "backchannel_first_token_logprob",
    "support_first_token_logprob",
    "wait_surface",
    "backchannel_surface",
    "support_surface",
    "hidden_norm",
    "injected_norm",
    "hidden_injected_cosine",
    "hook_calls",
    "seq_len",
    "injection_layer",
    "alpha",
    "position",
    "silence_elapsed",
    "delta_t",
    "utterance_elapsed",
    "is_user_speaking",
    "asr_changed",
]


def append_result(path, row):
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def load_result_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sequence_metrics_csv(rows):
    label_order = {"WAIT": 0, "BACKCHANNEL": 1, "SUPPORT": 2}
    by_context = defaultdict(list)
    for row in rows:
        by_context[row["context_id"]].append(row)
    total = correct = exact = regressions = 0
    premature = delayed = 0
    profile = defaultdict(lambda: {"total": 0, "correct": 0, "exact": 0, "contexts": 0})
    transitions = []
    for context_id, items in by_context.items():
        items = sorted(items, key=lambda x: float(x["silence_seconds"]))
        gold = [row["gold_label"] for row in items]
        pred = [row["generated_label"] for row in items]
        prof = items[0]["profile"]
        exact_match = int(gold == pred)
        exact += exact_match
        profile[prof]["exact"] += exact_match
        profile[prof]["contexts"] += 1
        transitions.append({"context_id": context_id, "profile": prof, "seconds": [float(x["silence_seconds"]) for x in items], "gold": gold, "pred": pred})
        for i, (g, pr) in enumerate(zip(gold, pred)):
            total += 1
            ok = int(g == pr)
            correct += ok
            profile[prof]["total"] += 1
            profile[prof]["correct"] += ok
            if label_order[pr] > label_order[g]:
                premature += 1
            if label_order[pr] < label_order[g]:
                delayed += 1
            if i and label_order[pred[i]] < label_order[pred[i - 1]]:
                regressions += 1
    context_total = max(len(by_context), 1)
    return {
        "step_accuracy": correct / max(total, 1),
        "exact_sequence_accuracy": exact / context_total,
        "premature_escalation_rate": premature / max(total, 1),
        "delayed_support_rate": delayed / max(total, 1),
        "regression_rate": regressions / max(total - context_total, 1),
        "profile_sequence_accuracy": {
            p: {
                "step_accuracy": v["correct"] / max(v["total"], 1),
                "exact_sequence_accuracy": v["exact"] / max(v["contexts"], 1),
                "contexts": v["contexts"],
            }
            for p, v in sorted(profile.items())
        },
        "transitions": transitions[:30],
    }


def summarize_results(result_path, run_dir):
    rows = load_result_rows(result_path)
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    metrics = {}
    failures = []
    for condition, items in sorted(by_condition.items()):
        y = [LABEL_TO_ID[row["gold_label"]] for row in items]
        pred = [LABEL_TO_ID[row["generated_label"]] for row in items]
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        seq = sequence_metrics_csv(items)
        proxy_agree = np.mean([row["generated_label"] == row["proxy_label"] for row in items]) if items else 0.0
        metrics[condition] = {
            "rows": len(items),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in range(3)},
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            "pred_counts": dict(Counter(row["generated_label"] for row in items)),
            "mean_probs": {label: float(np.mean([float(row[f"{label.lower()}_prob"]) for row in items])) for label in ["WAIT", "BACKCHANNEL", "SUPPORT"]},
            "mean_logprobs": {
                "WAIT": float(np.mean([float(row["wait_logprob"]) for row in items])),
                "BACKCHANNEL": float(np.mean([float(row["backchannel_logprob"]) for row in items])),
                "SUPPORT": float(np.mean([float(row["support_logprob"]) for row in items])),
            },
            "proxy_agreement": float(proxy_agree),
            "sequence": seq,
        }
        for row in items:
            if row["generated_label"] != row["gold_label"]:
                conf = max(float(row["wait_prob"]), float(row["backchannel_prob"]), float(row["support_prob"]))
                failures.append({**row, "confidence": conf})
    failures.sort(key=lambda x: x["confidence"], reverse=True)
    failure_path = run_dir / "failure_cases.csv"
    if failures:
        with failure_path.open("w", encoding="utf-8", newline="") as f:
            fields = list(failures[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(failures[:200])
    sequence_path = run_dir / "sequence_results.csv"
    with sequence_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["condition", "context_id", "profile", "seconds", "gold", "pred"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for condition, m in metrics.items():
            for tr in m["sequence"]["transitions"]:
                writer.writerow({"condition": condition, **tr})
    return metrics, failures[:50]


def plot_results(result_path, metrics, run_dir):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    run_plot_dir = run_dir / "plots"
    run_plot_dir.mkdir(exist_ok=True)
    rows = load_result_rows(result_path)
    conds = [c for c in CONDITIONS if c in metrics]
    plt.figure(figsize=(9, 4.5))
    plt.bar(conds, [metrics[c]["macro_f1"] for c in conds])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Macro F1")
    plt.title("Generation hook label macro F1 by condition")
    plt.tight_layout()
    plt.savefig(run_plot_dir / "ablation_macro_f1.png", dpi=170)
    plt.savefig(PLOT_DIR / f"{run_dir.name}_ablation_macro_f1.png", dpi=170)
    plt.close()

    if "correct_time_adapter" in metrics:
        cm = np.array(metrics["correct_time_adapter"]["confusion_matrix"])
        plt.figure(figsize=(4.4, 4.0))
        plt.imshow(cm, cmap="Blues")
        plt.xticks(range(3), LABELS, rotation=30, ha="right")
        plt.yticks(range(3), LABELS)
        for i in range(3):
            for j in range(3):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("Gold")
        plt.title("Correct Time Adapter confusion")
        plt.tight_layout()
        plt.savefig(run_plot_dir / "confusion_correct.png", dpi=170)
        plt.savefig(PLOT_DIR / f"{run_dir.name}_confusion_correct.png", dpi=170)
        plt.close()

    correct = [row for row in rows if row["condition"] == "correct_time_adapter"]
    if correct:
        profiles = sorted({row["profile"] for row in correct})
        fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)
        axes = axes.flatten()
        for ax, profile in zip(axes, profiles):
            prof_rows = [r for r in correct if r["profile"] == profile]
            times = sorted({float(r["silence_seconds"]) for r in prof_rows})
            for label, color in zip(LABELS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
                vals = []
                for t in times:
                    subset = [r for r in prof_rows if abs(float(r["silence_seconds"]) - t) < 1e-6]
                    vals.append(np.mean([float(r[f"{label.lower()}_prob"]) for r in subset]))
                ax.plot(times, vals, marker="o", label=label, color=color)
            ax.set_title(profile)
            ax.set_ylim(0, 1.02)
        for ax in axes[len(profiles):]:
            ax.axis("off")
        axes[0].legend(fontsize=7)
        fig.suptitle("Correct adapter probabilities by profile and time")
        fig.tight_layout()
        fig.savefig(run_plot_dir / "profile_time_probabilities_correct.png", dpi=170)
        fig.savefig(PLOT_DIR / f"{run_dir.name}_profile_time_probabilities_correct.png", dpi=170)
        plt.close(fig)

        plt.figure(figsize=(8, 4.5))
        label_order = {"WAIT": 0, "BACKCHANNEL": 1, "SUPPORT": 2}
        for profile in profiles:
            prof_rows = [r for r in correct if r["profile"] == profile]
            times = sorted({float(r["silence_seconds"]) for r in prof_rows})
            vals = []
            for t in times:
                subset = [r for r in prof_rows if abs(float(r["silence_seconds"]) - t) < 1e-6]
                vals.append(np.mean([label_order[r["generated_label"]] for r in subset]))
            plt.plot(times, vals, marker="o", label=profile)
        plt.yticks([0, 1, 2], LABELS)
        plt.xlabel("silence elapsed")
        plt.ylabel("mean predicted label")
        plt.title("Sequential transition graph")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(run_plot_dir / "sequence_transition_correct.png", dpi=170)
        plt.savefig(PLOT_DIR / f"{run_dir.name}_sequence_transition_correct.png", dpi=170)
        plt.close()


def zip_artifacts(run_dir):
    zip_path = run_dir / f"{run_dir.name}_artifacts.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in run_dir.rglob("*"):
            if path == zip_path or path.is_dir():
                continue
            zf.write(path, path.relative_to(run_dir))
    return zip_path


def run(args):
    set_seed(args.seed)
    run_dir = OUT_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    rows = load_split_rows(args.split, args.stage, args.max_contexts)
    if args.max_rows:
        rows = rows[: args.max_rows]
    vectors, no_time_hidden, artifact = build_vectors(rows, args.stage, args.layer)
    proxy = proxy_predictions(rows, vectors, no_time_hidden, artifact)
    processor, model, device, dtype = v1.load_model()
    model.disable_talker()
    model.eval()
    tokenizer = processor.tokenizer
    label_ids = make_label_candidates(tokenizer, args.label_surface)
    conditions = args.conditions.split(",") if args.conditions else CONDITIONS
    config = {
        "run_name": args.run_name,
        "model_id": MODEL_ID,
        "stage": args.stage,
        "split": args.split,
        "rows": len(rows),
        "contexts": len({row["context_id"] for row in rows}),
        "layer": args.layer,
        "alpha": args.alpha,
        "position": args.position,
        "prompt_style": args.prompt_style,
        "label_surface": args.label_surface,
        "conditions": conditions,
        "label_ids": label_ids,
        "prompt_has_seconds": False,
        "prompt": "No numeric seconds are included. Time enters only through injected vectors except oracle vectors derived from explicit hidden deltas.",
        "seed": args.seed,
    }
    write_json(run_dir / "config.json", config)
    result_path = run_dir / "per_condition_results.csv"
    logprob_path = run_dir / "per_timepoint_logprobs.csv"
    debug_path = run_dir / "hook_debug_log.txt"
    done = completed_keys(result_path)
    with debug_path.open("a", encoding="utf-8") as debug:
        debug.write(f"Run {args.run_name} started. rows={len(rows)} conditions={conditions}\n")
        debug.write(f"Label ids: {label_ids}\n")
    for row_i, row in enumerate(rows):
        inputs = build_inputs(processor, row, prompt_style=args.prompt_style)
        for condition in conditions:
            key = fkey(row, condition, args.layer, args.alpha, args.position)
            if key in done:
                continue
            idx = row_i
            vector = vectors[condition][idx]
            scores = score_labels(model, inputs, tokenizer, label_ids, vector, args.layer, args.alpha, args.position, condition, device, dtype)
            proxy_pred = LABELS[int(proxy[condition]["pred"][idx])]
            proxy_probs = proxy[condition]["probs"][idx]
            f = row["features"]
            out = {
                "key": key,
                "row_index": row_i,
                "id": row["id"],
                "context_id": row["context_id"],
                "profile": row["profile"],
                "fragment": row["fragment"],
                "silence_seconds": row["silence_seconds"],
                "gold_label": row["label"],
                "condition": condition,
                "proxy_label": proxy_pred,
                "proxy_wait_prob": float(proxy_probs[0]),
                "proxy_backchannel_prob": float(proxy_probs[1]),
                "proxy_support_prob": float(proxy_probs[2]),
                "injection_layer": args.layer,
                "alpha": args.alpha,
                "position": args.position,
                "silence_elapsed": f["silence_elapsed"],
                "delta_t": f["delta_t"],
                "utterance_elapsed": f["utterance_elapsed"],
                "is_user_speaking": f["is_user_speaking"],
                "asr_changed": f["asr_changed"],
                **scores,
            }
            append_result(result_path, out)
            append_result(logprob_path, out)
            done.add(key)
            if len(done) <= 30 or len(done) % 100 == 0:
                with debug_path.open("a", encoding="utf-8") as debug:
                    debug.write(
                        f"{len(done)} key={key} gold={row['label']} pred={scores['generated_label']} "
                        f"proxy={proxy_pred} hidden_norm={scores['hidden_norm']:.4f} inj_norm={scores['injected_norm']:.4f} calls={scores['hook_calls']}\n"
                    )
        if (row_i + 1) % 10 == 0 or row_i + 1 == len(rows):
            print(f"Progress rows {row_i + 1}/{len(rows)} completed_keys={len(done)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics, failures = summarize_results(result_path, run_dir)
    plot_results(result_path, metrics, run_dir)
    summary = {
        "config": config,
        "model_artifact": str(OUT_DIR / f"adapter_proxy_stage-{args.stage}_layer-{args.layer}.pt"),
        "metrics": metrics,
        "failures_top": failures,
        "artifacts": {
            "per_condition_results": str(result_path),
            "per_timepoint_logprobs": str(logprob_path),
            "sequence_results": str(run_dir / "sequence_results.csv"),
            "failure_cases": str(run_dir / "failure_cases.csv"),
            "hook_debug_log": str(debug_path),
            "plots": str(run_dir / "plots"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    zip_path = zip_artifacts(run_dir)
    summary["artifacts"]["zip"] = str(zip_path)
    write_json(run_dir / "summary.json", summary)
    print(f"Wrote {run_dir / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="sanity5")
    parser.add_argument("--stage", default="extra")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-contexts", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--position", default="last_token", choices=["last_token", "all_tokens"])
    parser.add_argument("--prompt-style", default="minimal", choices=["minimal", "defined"])
    parser.add_argument(
        "--label-surface",
        default="plain",
        choices=["plain", "leading_space", "newline", "best_plain_space", "best_plain_space_newline"],
    )
    parser.add_argument("--conditions", default="")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
