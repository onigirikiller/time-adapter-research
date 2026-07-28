from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
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
HIDDEN_DIR = ROOT / "artifacts/omni3b_sequential_v2/hidden_cache"
V3_ROOT = ROOT / "artifacts/omni3b_generation_hook_v3"
OUT_DIR = ROOT / "artifacts/omni3b_diagnostics_v4"
PLOT_DIR = OUT_DIR / "plots"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
SEED = 20260625


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1 = import_module(ROOT / "scripts/run_omni_sequential_time_adapter.py", "omni_v1_diag_v4")
v2 = import_module(ROOT / "scripts/run_omni3b_v2_experiment.py", "omni_v2_diag_v4")
v3 = import_module(ROOT / "scripts/run_omni3b_generation_hook_v3.py", "omni_v3_diag_v4")


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


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(split="test", stage="extra", max_contexts=0, max_rows=0):
    rows = [r for r in read_jsonl(DATA_DIR / f"{split}.jsonl") if stage in r["stages"]]
    if max_contexts:
        keep = []
        seen = set()
        for r in rows:
            if r["context_id"] not in seen:
                seen.add(r["context_id"])
                keep.append(r["context_id"])
            if len(keep) >= max_contexts:
                break
        keep = set(keep)
        rows = [r for r in rows if r["context_id"] in keep]
    if max_rows:
        rows = rows[:max_rows]
    return rows


def row_indices(rows, split="test", stage="extra"):
    full = [r for r in read_jsonl(DATA_DIR / f"{split}.jsonl") if stage in r["stages"]]
    idx = {r["id"]: i for i, r in enumerate(full)}
    return np.array([idx[r["id"]] for r in rows], dtype=np.int64)


def metric(rows, pred_labels, probs=None):
    y = np.array([LABEL_TO_ID[r["label"]] for r in rows], dtype=np.int64)
    p = np.array([LABEL_TO_ID[x] for x in pred_labels], dtype=np.int64)
    pr, re, f, s = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    out = {
        "rows": len(rows),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)),
        "backchannel_f1": float(f[LABEL_TO_ID["BACKCHANNEL"]]),
        "wait_recall": float(re[LABEL_TO_ID["WAIT"]]),
        "support_recall": float(re[LABEL_TO_ID["SUPPORT"]]),
        "per_class": {LABELS[i]: {"precision": float(pr[i]), "recall": float(re[i]), "f1": float(f[i]), "support": int(s[i])} for i in range(3)},
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
        "pred_counts": dict(Counter(pred_labels)),
    }
    if probs is not None:
        out["mean_probabilities"] = {LABELS[i]: float(np.mean(probs[:, i])) for i in range(3)}
    return out


def move_inputs(inputs, device, dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
        else:
            moved[key] = value
    return moved


def build_inputs_from_conv(processor, conv):
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    return processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)


def v3_conv(row, prompt_style="minimal", scheme=None):
    if scheme is None:
        text = v3.label_prompt(row, prompt_style=prompt_style)
    else:
        text = custom_label_prompt(row, scheme)
    return [
        {"role": "system", "content": [{"type": "text", "text": v3.SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "audio", "audio": str(Path(row["audio_path"]).resolve())}, {"type": "text", "text": text}]},
    ]


def capture_hidden_layers(processor, model, device, dtype, rows, prompt_kind="v3_minimal", mode="no_time", use_cache=None, cache_path=None):
    if cache_path and cache_path.exists():
        return np.load(cache_path)
    out = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            if prompt_kind == "v2":
                conv = v1.build_conversation(row, mode)
            elif prompt_kind == "v3_defined":
                conv = v3_conv(row, prompt_style="defined")
            else:
                conv = v3_conv(row, prompt_style="minimal")
            inputs = move_inputs(build_inputs_from_conv(processor, conv), device, dtype)
            kwargs = {"output_hidden_states": True, "use_audio_in_video": False}
            if use_cache is not None:
                kwargs["use_cache"] = use_cache
            outputs = model.thinker(**inputs, **kwargs)
            layers = [h[0, -1, :].detach().float().cpu().numpy().astype(np.float32) for h in outputs.hidden_states[1:]]
            out.append(np.stack(layers, axis=0))
            if (i + 1) % 25 == 0 or i == 0 or i + 1 == len(rows):
                print(f"hidden {prompt_kind}/{mode}: {i+1}/{len(rows)}", flush=True)
            del inputs, outputs
    arr = np.stack(out, axis=0)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
    return arr


def cosine_l2(a, b):
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.divide(np.sum(a * b, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
    l2 = np.linalg.norm(a - b, axis=1)
    return {
        "mean_cosine": float(np.mean(cos)),
        "median_cosine": float(np.median(cos)),
        "mean_l2": float(np.mean(l2)),
        "median_l2": float(np.median(l2)),
    }


def hidden_similarity(processor, model, device, dtype, rows):
    idx = row_indices(rows)
    cached_no = np.load(HIDDEN_DIR / "no_time_test.npy")[idx]
    cached_exp = np.load(HIDDEN_DIR / "explicit_test.npy")[idx]
    v2_live = capture_hidden_layers(processor, model, device, dtype, rows, prompt_kind="v2", mode="no_time", cache_path=OUT_DIR / "hidden_v2_live_no_time_subset.npy")
    v2_live_nocache = capture_hidden_layers(processor, model, device, dtype, rows, prompt_kind="v2", mode="no_time", use_cache=False, cache_path=OUT_DIR / "hidden_v2_live_no_time_use_cache_false_subset.npy")
    v3_min = capture_hidden_layers(processor, model, device, dtype, rows, prompt_kind="v3_minimal", mode="no_time", cache_path=OUT_DIR / "hidden_v3_minimal_no_time_subset.npy")
    v3_def = capture_hidden_layers(processor, model, device, dtype, rows, prompt_kind="v3_defined", mode="no_time", cache_path=OUT_DIR / "hidden_v3_defined_no_time_subset.npy")
    rows_out = []
    for layer in [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20]:
        pairs = {
            "live_v2_vs_cached_v2": (v2_live[:, layer, :], cached_no[:, layer, :]),
            "live_v2_use_cache_false_vs_cached_v2": (v2_live_nocache[:, layer, :], cached_no[:, layer, :]),
            "v3_minimal_vs_cached_v2_no_time": (v3_min[:, layer, :], cached_no[:, layer, :]),
            "v3_defined_vs_cached_v2_no_time": (v3_def[:, layer, :], cached_no[:, layer, :]),
            "cached_explicit_vs_cached_no_time": (cached_exp[:, layer, :], cached_no[:, layer, :]),
        }
        for name, (a, b) in pairs.items():
            rec = {"layer": layer, "comparison": name, **cosine_l2(a, b)}
            rows_out.append(rec)
    write_csv(OUT_DIR / "hidden_similarity.csv", rows_out)
    return rows_out


def decision_head_on_generation_hidden(rows, hidden_kind="v3_minimal"):
    idx = row_indices(rows)
    gen_hidden = np.load(OUT_DIR / f"hidden_{hidden_kind}_no_time_subset.npy")
    out_rows = []
    summary = {}
    for layer in [3, 8]:
        vectors, cached_no, artifact = v3.build_vectors(rows, "extra", layer)
        context_head = v2.Head(gen_hidden.shape[-1])
        context_head.load_state_dict(artifact["context_head_state"])
        context_head.eval()
        proxy_head = v2.Head(gen_hidden.shape[-1] * 2)
        proxy_head.load_state_dict(artifact["proxy_head_state"])
        proxy_head.eval()
        cond_features = {
            "time_adapter_decision_head_on_generation_hidden": v2.decision_features(gen_hidden[:, layer, :], vectors["correct_time_adapter"]),
            "oracle_delta_decision_head_on_generation_hidden": v2.decision_features(gen_hidden[:, layer, :], vectors["oracle_explicit_delta"]),
            "zero_delta_decision_head_on_generation_hidden": v2.decision_features(gen_hidden[:, layer, :], vectors["zero_vector"]),
            "shuffled_decision_head_on_generation_hidden": v2.decision_features(gen_hidden[:, layer, :], vectors["shuffled_time_adapter"]),
        }
        pred, probs = v2.predict(context_head, artifact["context_mean"], artifact["context_std"], gen_hidden[:, layer, :])
        labels = [LABELS[int(x)] for x in pred]
        summary[f"layer_{layer}_no_time_context_head_on_generation_hidden"] = metric(rows, labels, probs)
        for row, label, prob in zip(rows, labels, probs):
            out_rows.append({"layer": layer, "condition": "no_time_context_head_on_generation_hidden", "id": row["id"], "gold": row["label"], "pred": label, **{f"prob_{l}": float(prob[i]) for i, l in enumerate(LABELS)}})
        for name, feats in cond_features.items():
            pred, probs = v2.predict(proxy_head, artifact["proxy_mean"], artifact["proxy_std"], feats)
            labels = [LABELS[int(x)] for x in pred]
            summary[f"layer_{layer}_{name}"] = metric(rows, labels, probs)
            for row, label, prob in zip(rows, labels, probs):
                out_rows.append({"layer": layer, "condition": name, "id": row["id"], "gold": row["label"], "pred": label, **{f"prob_{l}": float(prob[i]) for i, l in enumerate(LABELS)}})
    write_csv(OUT_DIR / "generation_hidden_decision_head.csv", out_rows)
    return summary


LABEL_SCHEMES = {
    "long_en": {"WAIT": "WAIT", "BACKCHANNEL": "BACKCHANNEL", "SUPPORT": "SUPPORT"},
    "letters_wbs": {"WAIT": "W", "BACKCHANNEL": "B", "SUPPORT": "S"},
    "letters_abc": {"WAIT": "A", "BACKCHANNEL": "B", "SUPPORT": "C"},
    "angle_wbs": {"WAIT": "<W>", "BACKCHANNEL": "<B>", "SUPPORT": "<S>"},
    "semantic_short": {"WAIT": "WAIT", "BACKCHANNEL": "UH", "SUPPORT": "HELP"},
    "japanese": {"WAIT": "待つ", "BACKCHANNEL": "相槌", "SUPPORT": "支援"},
}


def custom_label_prompt(row, scheme_name):
    m = LABEL_SCHEMES[scheme_name]
    return (
        "Task: choose the listener timing label for a streaming dialogue system.\n"
        "Use the unfinished utterance, the audio context, and the hidden timer signal. Do not infer time from text.\n"
        f"Labels: {m['WAIT']} / {m['BACKCHANNEL']} / {m['SUPPORT']}\n"
        f"{m['WAIT']}: stay silent because the user is still speaking, repairing, or asked for time.\n"
        f"{m['BACKCHANNEL']}: give only a short acknowledgement without taking the turn.\n"
        f"{m['SUPPORT']}: take the turn with a brief supportive or answering response.\n"
        f"User fragment: \"{row['fragment']}\"\n"
        f"Answer with exactly one label: {m['WAIT']}, {m['BACKCHANNEL']}, or {m['SUPPORT']}.\n"
        "Label:"
    )


def label_candidates(tokenizer, scheme_name):
    out = {}
    for label, surface in LABEL_SCHEMES[scheme_name].items():
        variants = [surface, " " + surface, "\n" + surface]
        cands = []
        for v in variants:
            ids = tokenizer(v, add_special_tokens=False).input_ids
            if ids:
                cands.append({"surface": v, "ids": ids})
        out[label] = cands
    return out


def log_softmax_np(logits):
    arr = logits.detach().float().cpu().numpy().astype(np.float64)
    m = np.max(arr)
    return arr - (m + np.log(np.sum(np.exp(arr - m))))


class StepInjectionHook:
    def __init__(self, vector, layer_alpha=1.0, position="all_tokens", mode="add", target="layer"):
        self.vector = vector
        self.alpha = float(layer_alpha)
        self.position = position
        self.mode = mode
        self.target = target
        self.calls = 0
        self.stats = []

    def __call__(self, module, inputs, output):
        self.calls += 1
        if self.target == "lm_head_pre":
            hidden = inputs[0]
        else:
            hidden = output[0] if isinstance(output, tuple) else output
        vec = torch.as_tensor(self.vector, device=hidden.device, dtype=hidden.dtype).view(1, 1, -1) * self.alpha
        target_hidden = hidden[:, -1:, :]
        denom = torch.linalg.norm(target_hidden.detach().float().reshape(1, -1)) * torch.linalg.norm(vec.detach().float().reshape(1, -1))
        cosine = torch.sum(target_hidden.detach().float().reshape(1, -1) * vec.detach().float().reshape(1, -1)) / torch.clamp(denom, min=1e-12)
        self.stats.append({
            "call": self.calls,
            "seq_len": int(hidden.shape[1]),
            "hidden_norm": float(torch.linalg.norm(target_hidden.detach().float()).cpu()),
            "injected_norm": float(torch.linalg.norm(vec.detach().float()).cpu()),
            "cosine": float(cosine.cpu()),
        })
        if self.mode == "capture":
            return output
        if self.position == "all_tokens":
            modified = hidden + vec
        else:
            modified = hidden.clone()
            modified[:, -1:, :] = modified[:, -1:, :] + vec
        if self.target == "lm_head_pre":
            return (modified,) + tuple(inputs[1:])
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def pre(self, module, inputs):
        self.calls += 1
        hidden = inputs[0]
        vec = torch.as_tensor(self.vector, device=hidden.device, dtype=hidden.dtype).view(1, 1, -1) * self.alpha
        target_hidden = hidden[:, -1:, :]
        denom = torch.linalg.norm(target_hidden.detach().float().reshape(1, -1)) * torch.linalg.norm(vec.detach().float().reshape(1, -1))
        cosine = torch.sum(target_hidden.detach().float().reshape(1, -1) * vec.detach().float().reshape(1, -1)) / torch.clamp(denom, min=1e-12)
        self.stats.append({
            "call": self.calls,
            "seq_len": int(hidden.shape[1]),
            "hidden_norm": float(torch.linalg.norm(target_hidden.detach().float()).cpu()),
            "injected_norm": float(torch.linalg.norm(vec.detach().float()).cpu()),
            "cosine": float(cosine.cpu()),
        })
        if self.mode == "capture":
            return inputs
        modified = hidden.clone()
        modified[:, -1:, :] = modified[:, -1:, :] + vec
        return (modified,) + tuple(inputs[1:])


def next_logits_with_hook(model, inputs, append_ids, vector, layer, alpha, position, condition, device, dtype, target="layer", enable_hook=True):
    mode = "capture" if condition == "no_time" or not enable_hook else "add"
    hook = StepInjectionHook(vector, alpha, position, mode=mode, target=("lm_head_pre" if target == "final_hidden" else "layer"))
    if target == "final_hidden":
        handle = model.thinker.lm_head.register_forward_pre_hook(hook.pre)
    else:
        handle = model.thinker.model.layers[layer].register_forward_hook(hook)
    try:
        moved = v3.append_and_move(inputs, append_ids, device, dtype)
        with torch.inference_mode():
            outputs = model.thinker(**moved, use_audio_in_video=False)
        logits = outputs.logits[0, -1, :].detach().float().cpu()
    finally:
        handle.remove()
    stat = hook.stats[0] if hook.stats else {"call": 0, "seq_len": 0, "hidden_norm": 0.0, "injected_norm": 0.0, "cosine": 0.0}
    stat["hook_calls"] = hook.calls
    return logits, stat


def score_custom(model, inputs, label_ids, vector, layer, alpha, position, condition, device, dtype, target="layer", step_mode="every"):
    raw = {}
    avg = {}
    first = {}
    stats = None
    for label, cands in label_ids.items():
        best = None
        for cand in cands:
            ids = cand["ids"]
            logits, st = next_logits_with_hook(model, inputs, [], vector, layer, alpha, position, condition, device, dtype, target=target, enable_hook=True)
            if stats is None:
                stats = st
            lp = log_softmax_np(logits)
            total = float(lp[ids[0]])
            prefix = [ids[0]]
            for next_id in ids[1:]:
                enable = step_mode == "every"
                logits, _ = next_logits_with_hook(model, inputs, prefix, vector, layer, alpha, position, condition, device, dtype, target=target, enable_hook=enable)
                lp = log_softmax_np(logits)
                total += float(lp[next_id])
                prefix.append(next_id)
            item = {"raw": total, "avg": total / len(ids), "first": float(first.get(label, total if len(ids) == 1 else np.nan)), "surface": cand["surface"]}
            item["first"] = float(log_softmax_np(logits if len(ids) > 1 else next_logits_with_hook(model, inputs, [], vector, layer, alpha, position, condition, device, dtype, target=target, enable_hook=True)[0])[ids[0]]) if False else float(total if len(ids) == 1 else np.nan)
            if best is None or item["avg"] > best["avg"]:
                best = item
        raw[label] = best["raw"]
        avg[label] = best["avg"]
    vals = np.array([avg[l] for l in LABELS], dtype=np.float64)
    probs = np.exp(vals - np.max(vals))
    probs /= probs.sum()
    raw_vals = np.array([raw[l] for l in LABELS], dtype=np.float64)
    raw_probs = np.exp(raw_vals - np.max(raw_vals))
    raw_probs /= raw_probs.sum()
    return {
        "avg_pred": LABELS[int(np.argmax(probs))],
        "sum_pred": LABELS[int(np.argmax(raw_probs))],
        **{f"{l.lower()}_avg_logprob": float(avg[l]) for l in LABELS},
        **{f"{l.lower()}_sum_logprob": float(raw[l]) for l in LABELS},
        **{f"{l.lower()}_prob": float(probs[i]) for i, l in enumerate(LABELS)},
        **{f"{l.lower()}_raw_prob": float(raw_probs[i]) for i, l in enumerate(LABELS)},
        **(stats or {}),
    }


def label_surface_experiment(processor, model, device, dtype, rows):
    out = []
    summary = {}
    vectors, _, _ = v3.build_vectors(rows, "extra", 3)
    for scheme in LABEL_SCHEMES:
        ids = label_candidates(processor.tokenizer, scheme)
        no_time_scores = {}
        for i, row in enumerate(rows):
            inputs = build_inputs_from_conv(processor, v3_conv(row, scheme=scheme))
            for condition in ["no_time", "correct_time_adapter"]:
                vector = vectors[condition][i]
                t0 = time.perf_counter()
                score = score_custom(model, inputs, ids, vector, 3, 4.0, "all_tokens", condition, device, dtype)
                ms = (time.perf_counter() - t0) * 1000
                key = row["id"]
                if condition == "no_time":
                    no_time_scores[key] = score
                prior_vals = None
                prior_pred = ""
                if condition != "no_time" and key in no_time_scores:
                    prior_vals = np.array([score[f"{l.lower()}_avg_logprob"] - no_time_scores[key][f"{l.lower()}_avg_logprob"] for l in LABELS])
                    prior_pred = LABELS[int(np.argmax(prior_vals))]
                rec = {
                    "scheme": scheme,
                    "condition": condition,
                    "id": row["id"],
                    "profile": row["profile"],
                    "seconds": row["silence_seconds"],
                    "gold": row["label"],
                    "avg_pred": score["avg_pred"],
                    "sum_pred": score["sum_pred"],
                    "prior_delta_avg_pred": prior_pred,
                    "latency_ms": ms,
                    **{k: v for k, v in score.items() if k.endswith("_prob") or k.endswith("_logprob")},
                }
                out.append(rec)
            if (i + 1) % 20 == 0 or i == 0 or i + 1 == len(rows):
                print(f"label surfaces {scheme}: {i+1}/{len(rows)}", flush=True)
    write_csv(OUT_DIR / "label_surface_results.csv", out)
    for scheme in LABEL_SCHEMES:
        summary[scheme] = {}
        for condition in ["no_time", "correct_time_adapter"]:
            sub = [r for r in out if r["scheme"] == scheme and r["condition"] == condition]
            for pred_col in ["avg_pred", "sum_pred", "prior_delta_avg_pred"]:
                if pred_col == "prior_delta_avg_pred" and condition == "no_time":
                    continue
                pred = [r[pred_col] for r in sub]
                if any(not p for p in pred):
                    continue
                fake_rows = [{"label": r["gold"]} for r in sub]
                summary[scheme][f"{condition}_{pred_col}"] = metric(fake_rows, pred)
    return summary


def latency_experiment(processor, model, device, dtype, rows):
    vectors, _, _ = v3.build_vectors(rows, "extra", 3)
    label_ids = v3.make_label_candidates(processor.tokenizer, "plain")
    records = []
    for i, row in enumerate(rows):
        inputs = v3.build_inputs(processor, row, prompt_style="minimal")
        for condition in ["no_time", "correct_time_adapter"]:
            vector = vectors[condition][i]
            t0 = time.perf_counter()
            _ = v3.score_labels(model, inputs, processor.tokenizer, label_ids, vector, 3, 4.0, "all_tokens", condition, device, dtype)
            ms = (time.perf_counter() - t0) * 1000
            records.append({"row": i, "id": row["id"], "condition": condition, "latency_ms": ms})
    write_csv(OUT_DIR / "latency_results.csv", records)
    summary = {}
    for condition in ["no_time", "correct_time_adapter"]:
        vals = np.array([r["latency_ms"] for r in records if r["condition"] == condition], dtype=np.float64)
        summary[condition] = {
            "mean_ms": float(np.mean(vals)),
            "p50_ms": float(np.percentile(vals, 50)),
            "p90_ms": float(np.percentile(vals, 90)),
            "p95_ms": float(np.percentile(vals, 95)),
            "p99_ms": float(np.percentile(vals, 99)),
            "within_250ms": float(np.mean(vals <= 250)),
            "within_500ms": float(np.mean(vals <= 500)),
            "within_1000ms": float(np.mean(vals <= 1000)),
        }
        for period in [250, 500, 1000]:
            summary[condition][f"drift_sim_{period}ms"] = simulate_schedule(vals, period)
    return summary


def simulate_schedule(latencies_ms, period_ms):
    target = 0.0
    available = 0.0
    drifts = []
    skipped = 0
    backlog = 0
    for latency in latencies_ms:
        if available > target:
            backlog += 1
        start = max(target, available)
        end = start + latency
        drifts.append(start - target)
        available = end
        target += period_ms
    latest_only_available = 0.0
    latest_skipped = 0
    for latency in latencies_ms:
        if latest_only_available > target:
            latest_skipped += 1
            target += period_ms
            continue
        latest_only_available = target + latency
        target += period_ms
    return {
        "mean_start_drift_ms": float(np.mean(drifts)),
        "p95_start_drift_ms": float(np.percentile(drifts, 95)),
        "backlog_count": int(backlog),
        "latest_only_skipped_count": int(latest_skipped),
    }


def position_alpha_subset(processor, model, device, dtype, rows):
    vectors_by_layer = {layer: v3.build_vectors(rows, "extra", layer)[0] for layer in [3, 8]}
    label_ids = v3.make_label_candidates(processor.tokenizer, "plain")
    combos = []
    for layer in [3, 8]:
        for alpha in [1.0, 2.0, 4.0, 8.0]:
            for target, position in [("layer", "last_token"), ("layer", "all_tokens"), ("final_hidden", "last_token")]:
                for step_mode in ["first_only", "every"]:
                    combos.append((layer, alpha, target, position, step_mode))
    records = []
    for layer, alpha, target, position, step_mode in combos:
        preds = []
        stats = []
        for i, row in enumerate(rows):
            inputs = v3.build_inputs(processor, row, prompt_style="minimal")
            score = score_custom(model, inputs, label_ids, vectors_by_layer[layer]["correct_time_adapter"][i], layer, alpha, position, "correct_time_adapter", device, dtype, target=target, step_mode=step_mode)
            preds.append(score["avg_pred"])
            stats.append(score)
        m = metric(rows, preds)
        rec = {
            "layer": layer,
            "alpha": alpha,
            "target": target,
            "position": position,
            "step_mode": step_mode,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "backchannel_f1": m["backchannel_f1"],
            "wait_recall": m["wait_recall"],
            "support_recall": m["support_recall"],
            "pred_counts": json.dumps(m["pred_counts"], ensure_ascii=False),
            "mean_hidden_norm": float(np.mean([s["hidden_norm"] for s in stats])),
            "mean_injection_norm": float(np.mean([s["injected_norm"] for s in stats])),
            "mean_cosine": float(np.mean([s["cosine"] for s in stats])),
        }
        records.append(rec)
        print(f"sweep {rec}", flush=True)
    write_csv(OUT_DIR / "position_alpha_subset.csv", records)
    return records


def kv_cache_generate_probe(processor, model, device, dtype, rows):
    vectors, _, _ = v3.build_vectors(rows, "extra", 3)
    records = []
    for i, row in enumerate(rows[:5]):
        inputs = v3.build_inputs(processor, row, prompt_style="minimal")
        moved = move_inputs(inputs, device, dtype)
        for use_cache in [True, False]:
            for position in ["last_token", "all_tokens"]:
                hook = StepInjectionHook(vectors["correct_time_adapter"][i], 4.0, position=position, mode="add")
                handle = model.thinker.model.layers[3].register_forward_hook(hook)
                try:
                    with torch.inference_mode():
                        out = model.thinker.generate(**moved, max_new_tokens=3, do_sample=False, use_cache=use_cache, use_audio_in_video=False)
                    text = processor.tokenizer.decode(out[0][-3:], skip_special_tokens=False)
                    ok = True
                    err = ""
                except Exception as e:
                    text = ""
                    ok = False
                    err = repr(e)
                finally:
                    handle.remove()
                records.append({
                    "id": row["id"],
                    "gold": row["label"],
                    "use_cache": use_cache,
                    "position": position,
                    "ok": ok,
                    "error": err,
                    "generated_tail": text,
                    "hook_calls": hook.calls,
                    "seq_lens": json.dumps([s["seq_len"] for s in hook.stats]),
                    "inject_norms": json.dumps([s["injected_norm"] for s in hook.stats]),
                })
    write_csv(OUT_DIR / "kv_cache_hook_log.csv", records)
    return records


def audio_silence_check(rows):
    vals = []
    for row in rows:
        expected = float(row["silence_seconds"])
        speech = float(row["speech_duration_seconds"])
        total = float(row["total_duration_seconds"])
        vals.append(total - speech - expected)
    return {
        "rows": len(rows),
        "max_abs_total_minus_speech_minus_timer": float(np.max(np.abs(vals))),
        "mean_abs_total_minus_speech_minus_timer": float(np.mean(np.abs(vals))),
    }


def plot_outputs(summary):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    hs = summary.get("hidden_similarity", [])
    if hs:
        plt.figure(figsize=(9, 4.5))
        for comp in sorted({r["comparison"] for r in hs}):
            rows = [r for r in hs if r["comparison"] == comp]
            rows = sorted(rows, key=lambda x: x["layer"])
            plt.plot([r["layer"] for r in rows], [r["mean_cosine"] for r in rows], marker="o", label=comp)
        plt.xlabel("Layer")
        plt.ylabel("Mean cosine")
        plt.title("Proxy hidden vs generation prompt hidden")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "hidden_similarity_by_layer.png", dpi=180)
        plt.close()
    if (OUT_DIR / "label_surface_results.csv").exists():
        rows = []
        with (OUT_DIR / "label_surface_results.csv").open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        vals = []
        names = []
        for scheme in LABEL_SCHEMES:
            sub = [r for r in rows if r["scheme"] == scheme and r["condition"] == "correct_time_adapter"]
            if not sub:
                continue
            vals.append(metric([{"label": r["gold"]} for r in sub], [r["avg_pred"] for r in sub])["backchannel_f1"])
            names.append(scheme)
        plt.figure(figsize=(9, 4.5))
        plt.bar(names, vals)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("BACKCHANNEL F1")
        plt.title("Label surface comparison, correct adapter")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "label_surface_backchannel_f1.png", dpi=180)
        plt.close()
    if (OUT_DIR / "position_alpha_subset.csv").exists():
        rows = []
        with (OUT_DIR / "position_alpha_subset.csv").open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        plt.figure(figsize=(10, 5))
        labels = []
        vals = []
        for r in rows:
            labels.append(f"L{r['layer']} a{r['alpha']} {r['target']} {r['position']} {r['step_mode']}")
            vals.append(float(r["macro_f1"]))
        order = np.argsort(vals)[-20:]
        plt.barh([labels[i] for i in order], [vals[i] for i in order])
        plt.xlabel("Macro F1")
        plt.title("Top position/alpha subset runs")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "position_alpha_top20.png", dpi=180)
        plt.close()


def gpu_snapshot():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5)
        return out.strip()
    except Exception as e:
        return f"unavailable: {e!r}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-contexts", type=int, default=10)
    parser.add_argument("--label-contexts", type=int, default=8)
    parser.add_argument("--latency-contexts", type=int, default=5)
    parser.add_argument("--sweep-contexts", type=int, default=4)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-label-surface", action="store_true")
    args = parser.parse_args()

    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    processor, model, device, dtype = v1.load_model()
    model.disable_talker()
    model.eval()
    print(f"Loaded model on {device}; gpu={gpu_snapshot()}", flush=True)

    hidden_rows = load_rows(max_contexts=args.hidden_contexts)
    label_rows = load_rows(max_contexts=args.label_contexts)
    latency_rows = load_rows(max_contexts=args.latency_contexts)
    sweep_rows = load_rows(max_contexts=args.sweep_contexts)

    summary = {
        "model_id": v3.MODEL_ID,
        "gpu_start": gpu_snapshot(),
        "row_counts": {
            "hidden": len(hidden_rows),
            "label_surface": len(label_rows),
            "latency": len(latency_rows),
            "sweep": len(sweep_rows),
        },
    }
    summary["hidden_similarity"] = hidden_similarity(processor, model, device, dtype, hidden_rows)
    summary["generation_hidden_decision_head"] = decision_head_on_generation_hidden(hidden_rows)
    if not args.skip_label_surface:
        summary["label_surface"] = label_surface_experiment(processor, model, device, dtype, label_rows)
    summary["latency"] = latency_experiment(processor, model, device, dtype, latency_rows)
    if not args.skip_sweep:
        summary["position_alpha_subset"] = position_alpha_subset(processor, model, device, dtype, sweep_rows)
    summary["kv_cache_generate_probe"] = kv_cache_generate_probe(processor, model, device, dtype, sweep_rows[:5])
    summary["audio_silence_check"] = audio_silence_check(load_rows(max_contexts=50))
    summary["gpu_end"] = gpu_snapshot()
    plot_outputs(summary)
    summary["plots"] = {p.stem: str(p) for p in PLOT_DIR.glob("*.png")}
    write_json(OUT_DIR / "model_summary.json", summary)
    print(OUT_DIR / "model_summary.json", flush=True)


if __name__ == "__main__":
    main()
