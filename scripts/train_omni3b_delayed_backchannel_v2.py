from __future__ import annotations

import csv
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
SOURCE_DATA = ROOT / "data/omni3b_sequential_v2"
SOURCE_HIDDEN = ROOT / "artifacts/omni3b_sequential_v2/hidden_cache"
DATA_OUT = ROOT / "data/omni3b_delayed_backchannel_v2"
OUT_DIR = ROOT / "artifacts/omni3b_delayed_backchannel_tuning_v2"
FIG_DIR = ROOT / "output/figures/omni3b_delayed_backchannel_tuning_v2"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
SEED = 20260626
LAYER = 20


def import_v2():
    path = ROOT / "scripts/run_omni3b_v2_experiment.py"
    spec = importlib.util.spec_from_file_location("omni_v2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["omni_v2"] = module
    spec.loader.exec_module(module)
    return module


v2 = import_v2()


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dist(wait: float, back: float, support: float) -> list[float]:
    arr = np.asarray([wait, back, support], dtype=np.float32)
    arr = arr / max(float(arr.sum()), 1e-12)
    return [float(x) for x in arr]


def delayed_soft_label(profile: str, seconds: float) -> tuple[str, list[float], list[str]]:
    if profile == "asked_wait":
        if seconds < 3.0:
            return "WAIT", dist(0.96, 0.04, 0.0), ["WAIT"]
        return "WAIT", dist(0.86, 0.14, 0.0), ["WAIT", "BACKCHANNEL"]

    if profile == "self_repair":
        if seconds < 1.5:
            return "WAIT", dist(0.92, 0.08, 0.0), ["WAIT"]
        return "BACKCHANNEL", dist(0.28, 0.68, 0.04), ["WAIT", "BACKCHANNEL"]

    if profile == "finished":
        if seconds < 0.5:
            return "SUPPORT", dist(0.1, 0.1, 0.8), ["SUPPORT"]
        return "SUPPORT", dist(0.02, 0.08, 0.9), ["SUPPORT"]

    if profile == "direct_question":
        if seconds < 0.5:
            return "SUPPORT", dist(0.08, 0.12, 0.8), ["SUPPORT"]
        return "SUPPORT", dist(0.02, 0.08, 0.9), ["SUPPORT"]

    if profile == "vulnerable":
        if seconds < 0.75:
            return "WAIT", dist(0.88, 0.11, 0.01), ["WAIT", "BACKCHANNEL"]
        if seconds < 1.0:
            return "WAIT", dist(0.68, 0.30, 0.02), ["WAIT", "BACKCHANNEL"]
        if seconds < 4.0:
            return "BACKCHANNEL", dist(0.12, 0.76, 0.12), ["BACKCHANNEL", "SUPPORT"]
        return "SUPPORT", dist(0.03, 0.24, 0.73), ["BACKCHANNEL", "SUPPORT"]

    if profile == "neutral_incomplete":
        if seconds < 1.0:
            return "WAIT", dist(0.88, 0.11, 0.01), ["WAIT"]
        if seconds < 6.0:
            return "BACKCHANNEL", dist(0.18, 0.72, 0.10), ["WAIT", "BACKCHANNEL"]
        return "SUPPORT", dist(0.06, 0.36, 0.58), ["BACKCHANNEL", "SUPPORT"]

    if profile == "hesitant":
        if seconds < 1.0:
            return "WAIT", dist(0.88, 0.11, 0.01), ["WAIT", "BACKCHANNEL"]
        if seconds < 4.0:
            return "BACKCHANNEL", dist(0.12, 0.76, 0.12), ["BACKCHANNEL"]
        return "SUPPORT", dist(0.04, 0.31, 0.65), ["BACKCHANNEL", "SUPPORT"]

    if profile == "summary":
        if seconds < 1.0:
            return "WAIT", dist(0.82, 0.17, 0.01), ["WAIT", "BACKCHANNEL"]
        if seconds < 6.0:
            return "BACKCHANNEL", dist(0.13, 0.73, 0.14), ["BACKCHANNEL", "SUPPORT"]
        return "SUPPORT", dist(0.04, 0.31, 0.65), ["BACKCHANNEL", "SUPPORT"]

    raise ValueError(profile)


def relabel_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        label, soft, acceptable = delayed_soft_label(row["profile"], float(row["silence_seconds"]))
        new = dict(row)
        new["original_label"] = row["label"]
        new["label"] = label
        new["soft_label"] = {LABELS[i]: soft[i] for i in range(len(LABELS))}
        new["acceptable_labels"] = acceptable
        new["labeling_policy"] = "delayed_backchannel_v2_later_support_no_runtime_gate"
        out.append(new)
    return out


def y_labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([LABEL_TO_ID[row["label"]] for row in rows], dtype=np.int64)


def y_soft(rows: list[dict]) -> np.ndarray:
    return np.asarray([[row["soft_label"][label] for label in LABELS] for row in rows], dtype=np.float32)


def feature_matrix(rows: list[dict]) -> np.ndarray:
    vals = []
    for row in rows:
        f = row["features"]
        vals.append(
            [
                np.log1p(f["silence_elapsed"]),
                f["delta_t"],
                np.log1p(f["utterance_elapsed"]),
                1.0 if f["is_user_speaking"] else 0.0,
                1.0 if f["asr_changed"] else 0.0,
            ]
        )
    return np.asarray(vals, dtype=np.float32)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


class Head(nn.Module):
    def __init__(self, dim: int, hidden: int = 192, dropout: float = 0.12):
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


def decision_features(context: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return np.concatenate([context, delta], axis=1).astype(np.float32)


def predict(model: nn.Module, mean: np.ndarray, std: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    xs = torch.tensor(apply_standardizer(x, mean, std), dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(xs), dim=-1).cpu().numpy()
    return np.argmax(probs, axis=1), probs


def metric(rows: list[dict], pred: np.ndarray, probs: np.ndarray | None = None) -> dict:
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
        soft = y_soft(rows)
        out["brier"] = float(np.mean(np.sum((probs - soft) ** 2, axis=1)))
        out["mean_probabilities"] = {LABELS[i]: float(np.mean(probs[:, i])) for i in range(len(LABELS))}
    return out


def sequence_metrics(rows: list[dict], pred: np.ndarray) -> dict:
    by_context = defaultdict(list)
    for row, pr in zip(rows, pred):
        by_context[row["context_id"]].append((row, LABELS[int(pr)]))
    label_order = {"WAIT": 0, "BACKCHANNEL": 1, "SUPPORT": 2}
    exact = regressions = total_steps = correct_steps = 0
    first_backchannel = []
    first_support = []
    profile_counts = defaultdict(lambda: {"contexts": 0, "exact": 0, "steps": 0, "correct": 0})
    transitions = []
    for context_id, items in by_context.items():
        items = sorted(items, key=lambda x: x[0]["silence_seconds"])
        gold = [row["label"] for row, _ in items]
        got = [pr for _, pr in items]
        secs = [float(row["silence_seconds"]) for row, _ in items]
        prof = items[0][0]["profile"]
        step_correct = sum(g == p for g, p in zip(gold, got))
        exact_match = int(gold == got)
        exact += exact_match
        total_steps += len(items)
        correct_steps += step_correct
        profile_counts[prof]["contexts"] += 1
        profile_counts[prof]["exact"] += exact_match
        profile_counts[prof]["steps"] += len(items)
        profile_counts[prof]["correct"] += step_correct
        for i in range(1, len(got)):
            regressions += int(label_order[got[i]] < label_order[got[i - 1]])
        bc = next((s for s, label in zip(secs, got) if label == "BACKCHANNEL"), None)
        su = next((s for s, label in zip(secs, got) if label == "SUPPORT"), None)
        if bc is not None:
            first_backchannel.append(bc)
        if su is not None:
            first_support.append(su)
        transitions.append(
            {
                "context_id": context_id,
                "profile": prof,
                "fragment": items[0][0]["fragment"],
                "seconds": secs,
                "gold_sequence": gold,
                "prediction_sequence": got,
                "step_accuracy": float(step_correct / len(items)),
                "first_backchannel_s": bc,
                "first_support_s": su,
            }
        )
    return {
        "step_accuracy": float(correct_steps / total_steps),
        "exact_sequence_accuracy": float(exact / len(by_context)),
        "regression_rate": float(regressions / max(total_steps - len(by_context), 1)),
        "mean_first_backchannel_s": None if not first_backchannel else float(np.mean(first_backchannel)),
        "mean_first_support_s": None if not first_support else float(np.mean(first_support)),
        "profile": {
            p: {
                "step_accuracy": float(v["correct"] / v["steps"]),
                "exact_sequence_accuracy": float(v["exact"] / v["contexts"]),
                "contexts": int(v["contexts"]),
            }
            for p, v in sorted(profile_counts.items())
        },
        "transitions": transitions,
    }


def train_head(train_x: np.ndarray, train_soft: np.ndarray, eval_features: dict, eval_rows: dict, epochs: int = 220):
    set_seed()
    mean, std = fit_standardizer(train_x)
    x = torch.tensor(apply_standardizer(train_x, mean, std), dtype=torch.float32)
    y_soft_t = torch.tensor(train_soft, dtype=torch.float32)
    y_hard_t = torch.tensor(np.argmax(train_soft, axis=1), dtype=torch.long)
    model = Head(train_x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    rng = np.random.default_rng(SEED)
    curves = []
    best_state = None
    best_score = -1.0
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x))
        losses = []
        for start in range(0, len(order), 192):
            idx = order[start : start + 192]
            logits = model(x[idx])
            logp = nn.functional.log_softmax(logits, dim=-1)
            soft_loss = -(y_soft_t[idx] * logp).sum(dim=-1).mean()
            hard_loss = nn.functional.cross_entropy(logits, y_hard_t[idx])
            loss = 0.75 * soft_loss + 0.25 * hard_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % 2 == 0 or epoch == epochs - 1:
            row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            for cond, split_map in eval_features.items():
                for split in ["train", "validation", "test"]:
                    pred, probs = predict(model, mean, std, split_map[split])
                    m = metric(eval_rows[split], pred, probs)
                    row[f"{split}_{cond}_accuracy"] = m["accuracy"]
                    row[f"{split}_{cond}_macro_f1"] = m["macro_f1"]
                    row[f"{split}_{cond}_brier"] = m["brier"]
            score = row["validation_correct_time_adapter_macro_f1"] - 0.15 * row["validation_correct_time_adapter_brier"]
            curves.append(row)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, mean, std, {"curves": curves, "selected_epoch": best_epoch, "selection_score": best_score}


def adapter_predict(adapter: nn.Module, x: np.ndarray) -> np.ndarray:
    adapter.eval()
    with torch.no_grad():
        return adapter(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy().astype(np.float32)


def save_csv(path: Path, rows: list[dict]):
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_curves(curves: list[dict], path: Path):
    plt.figure(figsize=(9, 5))
    epochs = [r["epoch"] for r in curves]
    for cond in ["correct_time_adapter", "audio_only_zero", "oracle_explicit_delta"]:
        plt.plot(epochs, [r[f"validation_{cond}_macro_f1"] for r in curves], label=f"{cond} val")
        plt.plot(epochs, [r[f"test_{cond}_macro_f1"] for r in curves], linestyle="--", label=f"{cond} test")
    plt.xlabel("epoch")
    plt.ylabel("macro F1")
    plt.title("Delayed-backchannel tuning curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_transitions(transitions: list[dict], path: Path):
    chosen = []
    seen = set()
    for item in transitions:
        if item["profile"] not in seen:
            chosen.append(item)
            seen.add(item["profile"])
        if len(chosen) >= 8:
            break
    ymap = {label: i for i, label in enumerate(LABELS)}
    plt.figure(figsize=(9, 5))
    for tr in chosen:
        plt.plot(tr["seconds"], [ymap[x] for x in tr["prediction_sequence"]], marker="o", label=tr["profile"])
    plt.yticks(range(len(LABELS)), LABELS)
    plt.xlabel("silence_elapsed")
    plt.ylabel("predicted label")
    plt.title("Delayed model transitions by profile")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_prob_profile(rows: list[dict], probs: np.ndarray, path: Path):
    by_profile_second = defaultdict(list)
    for row, p in zip(rows, probs):
        by_profile_second[(row["profile"], float(row["silence_seconds"]))].append(p)
    profiles = sorted({r["profile"] for r in rows})
    seconds = sorted({float(r["silence_seconds"]) for r in rows})
    fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True, sharey=True)
    for ax, profile in zip(axes.ravel(), profiles):
        for label_i, label in enumerate(LABELS):
            vals = []
            for s in seconds:
                arr = np.asarray(by_profile_second[(profile, s)])
                vals.append(float(arr[:, label_i].mean()) if arr.size else np.nan)
            ax.plot(seconds, vals, marker="o", label=label)
        ax.set_title(profile)
        ax.set_ylim(-0.03, 1.03)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Mean prediction probability by profile and silence")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    set_seed()
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads((SOURCE_DATA / "manifest.json").read_text(encoding="utf-8"))
    splits = {split: relabel_rows(read_jsonl(SOURCE_DATA / f"{split}.jsonl")) for split in ["train", "validation", "test"]}
    for split, rows in splits.items():
        write_jsonl(DATA_OUT / f"{split}.jsonl", rows)

    manifest = {
        "created_at": "2026-06-26",
        "source_dataset": str(SOURCE_DATA),
        "source_hidden_cache": str(SOURCE_HIDDEN),
        "labeling_policy": "delayed_backchannel_v2_later_support",
        "train_validation_test_counts": {split: len(rows) for split, rows in splits.items()},
        "context_counts": {split: len({row["context_id"] for row in rows}) for split, rows in splits.items()},
        "label_counts": {split: dict(Counter(row["label"] for row in rows)) for split, rows in splits.items()},
        "original_label_counts": {split: dict(Counter(row["original_label"] for row in rows)) for split, rows in splits.items()},
        "profiles": source_manifest["profiles"],
        "time_points": source_manifest["time_points"],
        "note": "Runtime timing gates are not encoded here; later SUPPORT and thicker BACKCHANNEL bands are learned via hard and soft target labels.",
    }
    (DATA_OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    hidden = {
        mode: {split: np.load(SOURCE_HIDDEN / f"{mode}_{split}.npy", mmap_mode="r")[:, LAYER, :].astype(np.float32) for split in splits}
        for mode in ["audio_only", "explicit", "no_time"]
    }
    oracle_delta = {split: hidden["explicit"][split] - hidden["audio_only"][split] for split in splits}

    adapter = v2.v1.FeatureAdapter(5, 2048)
    adapter, adapter_summary = v2.v1.train_adapter(
        "delayed_audio_only_multi_feature",
        feature_matrix(splits["train"]),
        oracle_delta["train"],
        feature_matrix(splits["validation"]),
        oracle_delta["validation"],
        feature_matrix(splits["test"]),
        oracle_delta["test"],
        epochs=260,
    )
    correct_delta = {split: adapter_predict(adapter, feature_matrix(splits[split])) for split in splits}
    zero_delta = {split: np.zeros_like(correct_delta[split]) for split in splits}
    eval_features = {
        "correct_time_adapter": {split: decision_features(hidden["audio_only"][split], correct_delta[split]) for split in splits},
        "audio_only_zero": {split: decision_features(hidden["audio_only"][split], zero_delta[split]) for split in splits},
        "oracle_explicit_delta": {split: decision_features(hidden["audio_only"][split], oracle_delta[split]) for split in splits},
        "no_time_zero": {split: decision_features(hidden["no_time"][split], zero_delta[split]) for split in splits},
    }

    train_x = np.vstack([eval_features["correct_time_adapter"]["train"], eval_features["oracle_explicit_delta"]["train"]])
    train_soft = np.vstack([y_soft(splits["train"]), y_soft(splits["train"])])
    head, mean, std, head_summary = train_head(train_x, train_soft, eval_features, splits, epochs=220)

    results = {}
    predictions = {}
    probabilities = {}
    for cond, split_map in eval_features.items():
        results[cond] = {}
        for split in ["train", "validation", "test"]:
            pred, probs = predict(head, mean, std, split_map[split])
            results[cond][split] = metric(splits[split], pred, probs)
            if split == "test":
                predictions[cond] = pred
                probabilities[cond] = probs
        results[cond]["sequence"] = sequence_metrics(splits["test"], predictions[cond])

    curves_path = FIG_DIR / "delayed_v2_training_curves.png"
    transitions_path = FIG_DIR / "delayed_v2_test_transitions.png"
    prob_path = FIG_DIR / "delayed_v2_profile_probabilities.png"
    plot_curves(head_summary["curves"], curves_path)
    plot_transitions(results["correct_time_adapter"]["sequence"]["transitions"], transitions_path)
    plot_prob_profile(splits["test"], probabilities["correct_time_adapter"], prob_path)

    per_time_rows = []
    for row, pred_i, probs in zip(splits["test"], predictions["correct_time_adapter"], probabilities["correct_time_adapter"]):
        per_time_rows.append(
            {
                "context_id": row["context_id"],
                "profile": row["profile"],
                "seconds": row["silence_seconds"],
                "gold": row["label"],
                "pred": LABELS[int(pred_i)],
                "p_WAIT": float(probs[0]),
                "p_BACKCHANNEL": float(probs[1]),
                "p_SUPPORT": float(probs[2]),
                "fragment": row["fragment"],
            }
        )
    save_csv(OUT_DIR / "per_timepoint_test_predictions.csv", per_time_rows)

    checkpoint = {
        "adapter_state": adapter.state_dict(),
        "proxy_head_state": head.state_dict(),
        "proxy_mean": torch.as_tensor(mean),
        "proxy_std": torch.as_tensor(std),
        "layer": LAYER,
        "labels": LABELS,
        "model_id": "Qwen/Qwen2.5-Omni-3B",
        "base_hidden_mode": "audio_only",
        "adapter_target": "explicit_minus_audio_only",
        "labeling_policy": "delayed_backchannel_v2_later_support",
        "training_counts": manifest["train_validation_test_counts"],
        "selected_epoch": head_summary["selected_epoch"],
    }
    ckpt_path = OUT_DIR / "adapter_proxy_delayed_v2_audio_only_layer-20.pt"
    torch.save(checkpoint, ckpt_path)
    with (OUT_DIR / "heads_and_adapter.pkl").open("wb") as f:
        pickle.dump({"mean": mean, "std": std, "labels": LABELS}, f)

    summary = {
        "model_id": "Qwen/Qwen2.5-Omni-3B",
        "source_dataset": str(SOURCE_DATA),
        "derived_dataset": str(DATA_OUT),
        "hidden_cache_reused": str(SOURCE_HIDDEN),
        "layer": LAYER,
        "checkpoint": str(ckpt_path),
        "manifest": manifest,
        "adapter_summary": adapter_summary,
        "head_summary": head_summary,
        "metrics": results,
        "figures": {
            "curves": str(curves_path),
            "transitions": str(transitions_path),
            "profile_probabilities": str(prob_path),
        },
        "notes": [
            "No runtime minimum-silence gate is trained into the demo; later SUPPORT and delayed BACKCHANNEL behavior are represented by labels and soft labels.",
            "The original checkpoint was not overwritten.",
            "The source hidden cache is reused to avoid re-running thousands of Omni forwards; new demo forwards are run audio-only.",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(ckpt_path), "summary": str(OUT_DIR / "summary.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
