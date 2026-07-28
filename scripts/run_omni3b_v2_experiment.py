from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/omni3b_sequential_v2"
OUT_DIR = ROOT / "artifacts/omni3b_sequential_v2"
FIG_DIR = ROOT / "output/figures/omni3b_sequential_v2"
PREV_SUMMARY = ROOT / "artifacts/omni_sequential_time_adapter/summary.json"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
STAGES = ["small", "medium", "large", "extra"]
SEED = 20260623


def import_v1():
    path = ROOT / "scripts/run_omni_sequential_time_adapter.py"
    spec = importlib.util.spec_from_file_location("omni_v1", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["omni_v1"] = module
    spec.loader.exec_module(module)
    return module


v1 = import_v1()


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_data():
    splits = {split: read_jsonl(DATA_DIR / f"{split}.jsonl") for split in ["train", "validation", "test"]}
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    return splits, manifest


def extract_hidden_resumable(processor, model, device, dtype, rows: list[dict], mode: str, out_path: Path, numeric_note: str | None = None):
    if out_path.exists():
        return np.load(out_path)
    row_dir = out_path.with_suffix("")
    row_dir.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    with torch.inference_mode():
        for i, row in enumerate(rows):
            row_path = row_dir / f"{i:06d}.npy"
            if row_path.exists():
                continue
            conv = v1.build_conversation(row, mode, numeric_note=numeric_note)
            text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            audios, images, videos = v1.process_mm_info(conv, use_audio_in_video=False)
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = v1.move_inputs(inputs, device, dtype)
            outputs = model.thinker(**inputs, output_hidden_states=True, use_audio_in_video=False)
            layers = [h[0, -1, :].detach().float().cpu().numpy().astype(np.float32) for h in outputs.hidden_states[1:]]
            np.save(row_path, np.stack(layers, axis=0))
            if (i + 1) % 25 == 0 or i == 0 or i + 1 == total:
                print(f"Hidden progress {mode} {out_path.stem}: {i + 1}/{total}", flush=True)
            del inputs, outputs
            if torch.cuda.is_available() and (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
    hidden = np.stack([np.load(row_dir / f"{i:06d}.npy") for i in range(total)], axis=0)
    np.save(out_path, hidden)
    return hidden


def stage_indices(rows, stage):
    return np.array([i for i, row in enumerate(rows) if stage in row["stages"]], dtype=np.int64)


def slice_splits(splits, indices):
    return {split: [splits[split][int(i)] for i in idx] for split, idx in indices.items()}


def y_labels(rows):
    return np.array([LABEL_TO_ID[row["label"]] for row in rows], dtype=np.int64)


def feature_matrix(rows, kind="multi"):
    vals = []
    for row in rows:
        f = row["features"]
        if kind == "scalar":
            vals.append([np.log1p(f["silence_elapsed"])])
        else:
            vals.append([
                np.log1p(f["silence_elapsed"]),
                f["delta_t"],
                np.log1p(f["utterance_elapsed"]),
                1.0 if f["is_user_speaking"] else 0.0,
                1.0 if f["asr_changed"] else 0.0,
            ])
    return np.asarray(vals, dtype=np.float32)


def metric(rows, pred, probs=None):
    y = y_labels(rows)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(len(LABELS))), zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(LABELS)))).tolist(),
        "labels": LABELS,
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in range(len(LABELS))},
        "pred_counts": dict(Counter(LABELS[int(i)] for i in pred)),
    }
    if probs is not None:
        out["mean_probabilities"] = {LABELS[i]: float(np.mean(probs[:, i])) for i in range(len(LABELS))}
    return out


def fit_standardizer(x):
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_standardizer(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


class Head(nn.Module):
    def __init__(self, dim, hidden=192, dropout=0.12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(LABELS)),
        )

    def forward(self, x):
        return self.net(x)


def predict(model, mean, std, x):
    model.eval()
    xs = torch.tensor(apply_standardizer(x, mean, std), dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(xs), dim=-1).cpu().numpy()
    return np.argmax(probs, axis=1), probs


def train_head(name, train_x, train_y, eval_features, eval_rows, epochs=160, lr=8e-4):
    set_seed(SEED + train_x.shape[1])
    mean, std = fit_standardizer(train_x)
    x = torch.tensor(apply_standardizer(train_x, mean, std), dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.long)
    model = Head(train_x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    curves = []
    best_state = None
    best_score = -1.0
    best_epoch = 0
    rng = np.random.default_rng(SEED)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x))
        losses = []
        for start in range(0, len(order), 192):
            idx = order[start:start + 192]
            logits = model(x[idx])
            loss = nn.functional.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % 2 == 0 or epoch == epochs - 1:
            row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            for mode, split_map in eval_features.items():
                for split in ["train", "validation", "test"]:
                    pred, probs = predict(model, mean, std, split_map[split])
                    m = metric(eval_rows[split], pred, probs)
                    row[f"{split}_{mode}_accuracy"] = m["accuracy"]
                    row[f"{split}_{mode}_macro_f1"] = m["macro_f1"]
            score = row.get("validation_correct_time_adapter_macro_f1", row.get("validation_no_time_hidden_macro_f1", -1.0))
            curves.append(row)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, mean, std, {"name": name, "curves": curves, "selected_epoch": best_epoch, "selection_score": best_score}


def decision_features(context, delta):
    return np.concatenate([context, delta], axis=1).astype(np.float32)


def adapter_predict(adapter, x):
    adapter.eval()
    with torch.no_grad():
        return adapter(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy().astype(np.float32)


def shuffle_delta(delta, rows):
    rng = np.random.default_rng(SEED + len(rows))
    by_context = defaultdict(list)
    for i, row in enumerate(rows):
        by_context[row["context_id"]].append(i)
    out = np.zeros_like(delta)
    for idxs in by_context.values():
        shuffled = np.array(idxs, dtype=np.int64).copy()
        rng.shuffle(shuffled)
        out[np.array(idxs, dtype=np.int64)] = delta[shuffled]
    return out


def random_norm_delta(delta):
    rng = np.random.default_rng(SEED + delta.shape[0])
    direction = rng.normal(size=delta.shape).astype(np.float32)
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    return direction * norms


def sequence_metrics(rows, pred):
    by_context = defaultdict(list)
    for row, pr in zip(rows, pred):
        by_context[row["context_id"]].append((row, LABELS[int(pr)]))
    label_order = {"WAIT": 0, "BACKCHANNEL": 1, "SUPPORT": 2}
    total = correct = exact = regressions = 0
    premature = delayed = 0
    finished_short_total = finished_short_support = 0
    suppress_total = suppress_ok = 0
    profile = defaultdict(lambda: {"total": 0, "correct": 0, "exact": 0, "contexts": 0})
    transitions = []
    for context_id, items in by_context.items():
        items = sorted(items, key=lambda x: x[0]["silence_seconds"])
        gold = [row["label"] for row, _ in items]
        got = [pr for _, pr in items]
        secs = [row["silence_seconds"] for row, _ in items]
        prof = items[0][0]["profile"]
        step_correct = sum(g == p for g, p in zip(gold, got))
        total += len(items)
        correct += step_correct
        exact += int(gold == got)
        profile[prof]["total"] += len(items)
        profile[prof]["correct"] += step_correct
        profile[prof]["exact"] += int(gold == got)
        profile[prof]["contexts"] += 1
        for i in range(1, len(got)):
            regressions += int(label_order[got[i]] < label_order[got[i - 1]])
        for row, pr in items:
            if pr == "SUPPORT" and row["label"] != "SUPPORT":
                premature += 1
            if row["label"] == "SUPPORT" and pr != "SUPPORT":
                delayed += 1
            if row["profile"] in {"finished", "direct_question"} and row["silence_seconds"] <= 0.25:
                finished_short_total += 1
                finished_short_support += int(pr == "SUPPORT")
            if row["profile"] in {"asked_wait", "self_repair"}:
                suppress_total += 1
                suppress_ok += int(pr != "SUPPORT")
        transitions.append({
            "context_id": context_id,
            "profile": prof,
            "fragment": items[0][0]["fragment"],
            "seconds": secs,
            "gold_sequence": gold,
            "prediction_sequence": got,
            "step_accuracy": float(step_correct / len(items)),
        })
    return {
        "step_accuracy": float(correct / total),
        "exact_sequence_accuracy": float(exact / len(by_context)),
        "premature_escalation_rate": float(premature / total),
        "delayed_support_rate": float(delayed / total),
        "regression_rate": float(regressions / max(total - len(by_context), 1)),
        "finished_direct_question_short_support_rate": float(finished_short_support / max(finished_short_total, 1)),
        "asked_wait_self_repair_support_suppression_rate": float(suppress_ok / max(suppress_total, 1)),
        "profile": {
            p: {
                "step_accuracy": float(v["correct"] / v["total"]),
                "exact_sequence_accuracy": float(v["exact"] / v["contexts"]),
                "contexts": int(v["contexts"]),
            }
            for p, v in sorted(profile.items())
        },
        "transitions": transitions,
    }


def profile_accuracy(rows, pred):
    by_profile = defaultdict(lambda: {"correct": 0, "total": 0})
    for row, pr in zip(rows, pred):
        p = row["profile"]
        by_profile[p]["total"] += 1
        by_profile[p]["correct"] += int(LABELS[int(pr)] == row["label"])
    return {p: float(v["correct"] / v["total"]) for p, v in sorted(by_profile.items())}


def grid(rows, probs, pred):
    out = []
    for row, p, pr in zip(rows, probs, pred):
        out.append({
            "context_id": row["context_id"],
            "profile": row["profile"],
            "fragment": row["fragment"],
            "seconds": row["silence_seconds"],
            "gold": row["label"],
            "prediction": LABELS[int(pr)],
            "probabilities": {LABELS[i]: float(p[i]) for i in range(len(LABELS))},
            "support_minus_wait": float(p[LABEL_TO_ID["SUPPORT"]] - p[LABEL_TO_ID["WAIT"]]),
            "backchannel_minus_wait": float(p[LABEL_TO_ID["BACKCHANNEL"]] - p[LABEL_TO_ID["WAIT"]]),
        })
    return out


def failures(rows, pred, probs, limit=30):
    vals = []
    for row, pr, p in zip(rows, pred, probs):
        label = LABELS[int(pr)]
        if label != row["label"]:
            vals.append({
                "context_id": row["context_id"],
                "profile": row["profile"],
                "seconds": row["silence_seconds"],
                "fragment": row["fragment"],
                "gold": row["label"],
                "prediction": label,
                "confidence": float(np.max(p)),
                "probabilities": {LABELS[i]: float(p[i]) for i in range(len(LABELS))},
            })
    vals.sort(key=lambda x: x["confidence"], reverse=True)
    return vals[:limit]


def plot_stage_bars(stage_results, path):
    names = list(stage_results)
    conditions = ["no_time_hidden", "zero_vector", "correct_time_adapter", "shuffled_time_adapter", "random_norm_matched", "non_time_numeric", "oracle_explicit_delta"]
    x = np.arange(len(names))
    width = 0.105
    plt.figure(figsize=(11, 5))
    for i, cond in enumerate(conditions):
        vals = [stage_results[s]["metrics"][cond]["test"]["macro_f1"] for s in names]
        plt.bar(x + (i - 3) * width, vals, width, label=cond)
    plt.xticks(x, names)
    plt.ylabel("Test macro F1")
    plt.title("Omni3B v2 ablation by dataset scale")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_curves(curves, path, title):
    plt.figure(figsize=(9, 4.8))
    epochs = [r["epoch"] for r in curves]
    for cond in ["no_time_hidden", "correct_time_adapter", "oracle_explicit_delta"]:
        if f"validation_{cond}_macro_f1" not in curves[0]:
            continue
        plt.plot(epochs, [r[f"validation_{cond}_macro_f1"] for r in curves], label=f"{cond} val")
        plt.plot(epochs, [r[f"test_{cond}_macro_f1"] for r in curves], linestyle="--", label=f"{cond} test")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_transitions(transitions, path, title, max_profiles=8):
    chosen = []
    seen = set()
    for tr in transitions:
        if tr["profile"] not in seen:
            chosen.append(tr)
            seen.add(tr["profile"])
        if len(chosen) >= max_profiles:
            break
    ymap = {label: i for i, label in enumerate(LABELS)}
    plt.figure(figsize=(9, 5))
    for tr in chosen:
        plt.plot(tr["seconds"], [ymap[x] for x in tr["prediction_sequence"]], marker="o", label=tr["profile"])
    plt.yticks(range(len(LABELS)), LABELS)
    plt.xlabel("silence_elapsed")
    plt.ylabel("Predicted label")
    plt.title(title)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_probability_grid(grid_items, path):
    chosen = []
    seen = set()
    for item in grid_items:
        cid = item["context_id"]
        if cid not in seen:
            chosen.append(cid)
            seen.add(cid)
        if len(chosen) >= 8:
            break
    by = defaultdict(list)
    for item in grid_items:
        if item["context_id"] in chosen:
            by[item["context_id"]].append(item)
    fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True, sharey=True)
    for ax, (cid, items) in zip(axes.ravel(), by.items()):
        items = sorted(items, key=lambda x: x["seconds"])
        xs = [x["seconds"] for x in items]
        for label in LABELS:
            ax.plot(xs, [x["probabilities"][label] for x in items], marker="o", label=label)
        ax.set_title(items[0]["profile"], fontsize=9)
        ax.set_ylim(-0.03, 1.03)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Same utterance, changing time only: prediction probabilities")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(path, dpi=180)
    plt.close()


def plot_confusion(cm, path, title):
    arr = np.asarray(cm)
    plt.figure(figsize=(4.8, 4.2))
    im = plt.imshow(arr, cmap="Blues")
    plt.colorbar(im)
    plt.xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    plt.yticks(range(len(LABELS)), LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, str(arr[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hidden_dir = OUT_DIR / "hidden_cache"
    hidden_dir.mkdir(exist_ok=True)
    splits, manifest = load_data()
    processor, model, device, dtype = v1.load_model()
    print(f"Loaded {v1.MODEL_ID} on {device}", flush=True)

    hidden = {mode: {} for mode in ["audio_only", "no_time", "explicit"]}
    for mode in hidden:
        for split, rows in splits.items():
            out_path = hidden_dir / f"{mode}_{split}.npy"
            hidden[mode][split] = extract_hidden_resumable(processor, model, device, dtype, rows, mode, out_path)
            print(f"Hidden {mode} {split}: {hidden[mode][split].shape}", flush=True)

    probe_rows = splits["train"][:32]
    probe_numeric = {}
    for name, note in {"kg": "[5kg]", "m": "[5m]", "score": "[5 points]", "yen": "[5 yen]"}.items():
        probe_numeric[name] = extract_hidden_resumable(processor, model, device, dtype, probe_rows, "no_time", out_path=hidden_dir / f"numeric_probe_{name}.npy", numeric_note=note)
        print(f"Numeric probe {name}: {probe_numeric[name].shape}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    full_indices = {stage: {split: stage_indices(splits[split], stage) for split in ["train", "validation", "test"]} for stage in STAGES}
    stage_results = {}
    figures = {"ablation_by_stage": str(FIG_DIR / "ablation_by_stage.png")}
    for stage in STAGES:
        print(f"Stage {stage}", flush=True)
        idxs = full_indices[stage]
        stage_rows = slice_splits(splits, idxs)
        stage_hidden = {
            mode: {split: hidden[mode][split][idxs[split]] for split in ["train", "validation", "test"]}
            for mode in hidden
        }
        layer, layer_metrics = v1.select_layer(stage_hidden["explicit"], stage_rows)
        time_prediction = v1.time_prediction_metrics(stage_hidden["explicit"], stage_rows, layer)
        no_time = {split: stage_hidden["no_time"][split][:, layer, :] for split in stage_rows}
        audio_only = {split: stage_hidden["audio_only"][split][:, layer, :] for split in stage_rows}
        explicit = {split: stage_hidden["explicit"][split][:, layer, :] for split in stage_rows}
        oracle_delta = {split: explicit[split] - no_time[split] for split in stage_rows}

        adapter, adapter_summary = v1.train_adapter(
            "multi_feature",
            feature_matrix(stage_rows["train"], "multi"), oracle_delta["train"],
            feature_matrix(stage_rows["validation"], "multi"), oracle_delta["validation"],
            feature_matrix(stage_rows["test"], "multi"), oracle_delta["test"],
            epochs=260,
        )
        correct_delta = {split: adapter_predict(adapter, feature_matrix(stage_rows[split], "multi")) for split in stage_rows}
        zero_delta = {split: np.zeros_like(correct_delta[split]) for split in stage_rows}
        shuffled_delta = {split: shuffle_delta(correct_delta[split], stage_rows[split]) for split in stage_rows}
        random_delta = {split: random_norm_delta(correct_delta[split]) for split in stage_rows}

        non_time_delta = {}
        non_time_report = {}
        mean_adapter = np.mean(correct_delta["train"], axis=0)
        for name, probe in probe_numeric.items():
            direction = np.mean(probe[:, layer, :] - hidden["no_time"]["train"][: len(probe_rows), layer, :], axis=0)
            direction = direction / max(np.linalg.norm(direction), 1e-12)
            non_time_delta[name] = {}
            for split in stage_rows:
                norms = np.linalg.norm(correct_delta[split], axis=1, keepdims=True)
                non_time_delta[name][split] = direction.reshape(1, -1) * norms
            denom = max(np.linalg.norm(mean_adapter), 1e-12)
            non_time_report[name] = {"cosine_to_adapter_mean": float(np.dot(direction, mean_adapter) / denom)}
        chosen_non_time = "score"

        adapter_eval_features = {
            "correct_time_adapter": {split: decision_features(no_time[split], correct_delta[split]) for split in stage_rows},
            "shuffled_time_adapter": {split: decision_features(no_time[split], shuffled_delta[split]) for split in stage_rows},
            "zero_vector": {split: decision_features(no_time[split], zero_delta[split]) for split in stage_rows},
            "random_norm_matched": {split: decision_features(no_time[split], random_delta[split]) for split in stage_rows},
            "non_time_numeric": {split: decision_features(no_time[split], non_time_delta[chosen_non_time][split]) for split in stage_rows},
            "oracle_explicit_delta": {split: decision_features(no_time[split], oracle_delta[split]) for split in stage_rows},
            "audio_only_zero": {split: decision_features(audio_only[split], zero_delta[split]) for split in stage_rows},
        }
        train_x = np.vstack([adapter_eval_features["correct_time_adapter"]["train"], adapter_eval_features["oracle_explicit_delta"]["train"]])
        train_y = np.concatenate([y_labels(stage_rows["train"]), y_labels(stage_rows["train"])])
        adapter_head, adapter_mean, adapter_std, adapter_head_summary = train_head(
            "adapter_head",
            train_x,
            train_y,
            adapter_eval_features,
            stage_rows,
            epochs=170,
        )

        no_time_features = {"no_time_hidden": {split: no_time[split] for split in stage_rows}}
        context_head, context_mean, context_std, context_summary = train_head(
            "no_time_context_head",
            no_time["train"],
            y_labels(stage_rows["train"]),
            no_time_features,
            stage_rows,
            epochs=170,
            lr=6e-4,
        )

        metrics = {}
        predictions = {}
        probabilities = {}
        pred, probs = predict(context_head, context_mean, context_std, no_time["test"])
        metrics["no_time_hidden"] = {"test": metric(stage_rows["test"], pred, probs)}
        predictions["no_time_hidden"] = pred
        probabilities["no_time_hidden"] = probs
        for cond, feat in adapter_eval_features.items():
            metrics[cond] = {}
            for split in ["train", "validation", "test"]:
                pred, probs = predict(adapter_head, adapter_mean, adapter_std, feat[split])
                metrics[cond][split] = metric(stage_rows[split], pred, probs)
                if split == "test":
                    predictions[cond] = pred
                    probabilities[cond] = probs
        for name, deltas in non_time_delta.items():
            feats = {split: decision_features(no_time[split], deltas[split]) for split in stage_rows}
            pred, probs = predict(adapter_head, adapter_mean, adapter_std, feats["test"])
            non_time_report[name]["metrics"] = metric(stage_rows["test"], pred, probs)

        for cond, pred in predictions.items():
            if cond not in metrics:
                continue
            metrics[cond]["profile_accuracy"] = profile_accuracy(stage_rows["test"], pred)
            metrics[cond]["sequence"] = sequence_metrics(stage_rows["test"], pred)

        adapter_grid = grid(stage_rows["test"], probabilities["correct_time_adapter"], predictions["correct_time_adapter"])
        oracle_grid = grid(stage_rows["test"], probabilities["oracle_explicit_delta"], predictions["oracle_explicit_delta"])
        stage_figs = {
            "curves": str(FIG_DIR / f"{stage}_curves.png"),
            "transitions": str(FIG_DIR / f"{stage}_transitions.png"),
            "probabilities": str(FIG_DIR / f"{stage}_probabilities.png"),
            "confusion_correct": str(FIG_DIR / f"{stage}_confusion_correct.png"),
        }
        plot_curves(adapter_head_summary["curves"], Path(stage_figs["curves"]), f"{stage} adapter head curves")
        plot_transitions(metrics["correct_time_adapter"]["sequence"]["transitions"], Path(stage_figs["transitions"]), f"{stage} correct adapter transitions")
        plot_probability_grid(adapter_grid, Path(stage_figs["probabilities"]))
        plot_confusion(metrics["correct_time_adapter"]["test"]["confusion_matrix"], Path(stage_figs["confusion_correct"]), f"{stage} correct adapter test")

        train_f1 = adapter_head_summary["curves"][-1]["train_correct_time_adapter_macro_f1"]
        val_best = adapter_head_summary["selection_score"]
        test_best = metrics["correct_time_adapter"]["test"]["macro_f1"]
        overfitting = {
            "train_final_correct_f1": float(train_f1),
            "best_validation_correct_f1": float(val_best),
            "test_correct_f1": float(test_best),
            "train_validation_gap": float(train_f1 - val_best),
            "flag": bool((train_f1 - val_best) > 0.15 and test_best < val_best + 0.02),
        }
        stage_results[stage] = {
            "stage": stage,
            "selected_layer": int(layer),
            "layer_selection_metrics": layer_metrics,
            "time_prediction": time_prediction,
            "adapter": adapter_summary,
            "adapter_head": adapter_head_summary,
            "context_head": context_summary,
            "metrics": metrics,
            "non_time_numeric": non_time_report,
            "context_x_time": {"correct_adapter": adapter_grid, "oracle_explicit_delta": oracle_grid},
            "failures": {
                cond: failures(stage_rows["test"], predictions[cond], probabilities[cond])
                for cond in ["correct_time_adapter", "oracle_explicit_delta", "zero_vector", "shuffled_time_adapter", "no_time_hidden"]
                if cond in predictions
            },
            "overfitting": overfitting,
            "figures": stage_figs,
            "counts": {
                split: {
                    "rows": len(stage_rows[split]),
                    "contexts": len({row["context_id"] for row in stage_rows[split]}),
                    "labels": dict(Counter(row["label"] for row in stage_rows[split])),
                    "profiles": dict(Counter(row["profile"] for row in stage_rows[split])),
                }
                for split in stage_rows
            },
        }

    plot_stage_bars(stage_results, Path(figures["ablation_by_stage"]))
    prev = json.loads(PREV_SUMMARY.read_text(encoding="utf-8")) if PREV_SUMMARY.exists() else None
    summary = {
        "model_id": v1.MODEL_ID,
        "dataset": str(DATA_DIR),
        "dataset_manifest": manifest,
        "previous_v1": {
            "adapter_multi_test_accuracy": prev["metrics"]["adapter_multi"]["test"]["accuracy"],
            "adapter_multi_test_macro_f1": prev["metrics"]["adapter_multi"]["test"]["macro_f1"],
            "selected_layer": prev["selected_layer"],
        } if prev else None,
        "stage_results": stage_results,
        "figures": figures,
        "artifacts": {
            "hidden_cache": str(hidden_dir),
            "summary": str(OUT_DIR / "summary.json"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "heads.pkl").open("wb") as f:
        pickle.dump({"note": "Stage-specific heads are summarized in JSON; weights are not retained to keep the artifact compact."}, f)
    print(f"Wrote {OUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
