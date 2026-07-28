from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATA_DIR = ROOT / "data/qwen3_context_time_phase1_3000"
OUT_DIR = ROOT / "artifacts/qwen3_phase1_3000"
FIG_DIR = ROOT / "output/figures/qwen3_phase1_3000"
CACHE_DIR = ROOT / ".cache/huggingface"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
SEED = 20260623


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dataset() -> tuple[dict[str, list[dict]], dict]:
    splits = {split: read_jsonl(DATA_DIR / f"{split}.jsonl") for split in ["train", "validation", "test"]}
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    return splits, manifest


def build_prompt(row: dict, include_timing: bool = True, override_control: str | None = None) -> str:
    timing = row["time_expression"] if include_timing else "not provided"
    control = override_control
    if control is None and row.get("has_negative_control"):
        control = row.get("unrelated_numeric_note")
    control_line = f"Unrelated numeric note: {control}\n" if control else ""
    return (
        "Task: choose the listener timing label for a streaming dialogue system.\n"
        "Use both the unfinished utterance and the timing cue. If an unrelated numeric note is present, do not treat it as time.\n"
        "Labels:\n"
        "WAIT = keep listening because the speaker likely wants to continue.\n"
        "BACKCHANNEL = give a short acknowledgement or small prompt without taking over.\n"
        "SUPPORT = actively respond, answer, or offer gentle help.\n"
        f"User fragment: \"{row['fragment']}\"\n"
        f"Timing cue: {timing}\n"
        f"{control_line}"
        "Answer with exactly one label.\n"
        "Label:"
    )


def load_model(cache_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(cache_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=str(cache_dir), dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=str(cache_dir), torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def extract_hidden(tokenizer, model, device: torch.device, prompts: list[str], out_path: Path | None, batch_size: int) -> np.ndarray:
    if out_path is not None and out_path.exists():
        return np.load(out_path)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            lengths = inputs["attention_mask"].sum(dim=1) - 1
            outputs = model(**inputs, output_hidden_states=True)
            batch_idx = torch.arange(len(batch), device=device)
            layers = []
            for hidden in outputs.hidden_states[1:]:
                selected = hidden[batch_idx, lengths, :].detach().float().cpu().numpy().astype(np.float32)
                layers.append(selected)
            rows.append(np.stack(layers, axis=1))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    hidden = np.concatenate(rows, axis=0)
    if out_path is not None:
        np.save(out_path, hidden)
    return hidden


def y_seconds(rows: list[dict]) -> np.ndarray:
    return np.array([row["seconds"] for row in rows], dtype=np.float32)


def y_labels(rows: list[dict]) -> np.ndarray:
    return np.array([LABEL_TO_ID[row["label"]] for row in rows], dtype=np.int64)


def y_soft(rows: list[dict]) -> np.ndarray:
    return np.array([[row["soft_label"][label] for label in LABELS] for row in rows], dtype=np.float32)


def time_metrics_for_layer(x_train: np.ndarray, s_train: np.ndarray, x_eval: np.ndarray, s_eval: np.ndarray):
    y_train = np.log1p(s_train)
    y_eval = np.log1p(s_eval)
    reg = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    reg.fit(x_train, y_train)
    pred = reg.predict(x_eval)
    corr = float(np.corrcoef(y_eval, pred)[0, 1]) if len(set(np.round(y_eval, 6))) > 1 else 0.0
    cls_train = (s_train >= 2.0).astype(np.int64)
    cls_eval = (s_eval >= 2.0).astype(np.int64)
    clf = Pipeline([("scale", StandardScaler()), ("logreg", LogisticRegression(C=0.5, max_iter=1000, solver="liblinear"))])
    clf.fit(x_train, cls_train)
    cls_pred = clf.predict(x_eval)
    return {
        "r2_log_seconds": float(r2_score(y_eval, pred)),
        "corr_log_seconds": corr,
        "long_pause_accuracy": float(accuracy_score(cls_eval, cls_pred)),
    }


def select_time_layer(explicit_hidden: dict[str, np.ndarray], splits: dict[str, list[dict]]) -> tuple[int, list[dict]]:
    s_train = y_seconds(splits["train"])
    s_val = y_seconds(splits["validation"])
    metrics = []
    for layer in range(explicit_hidden["train"].shape[1]):
        m = time_metrics_for_layer(explicit_hidden["train"][:, layer, :], s_train, explicit_hidden["validation"][:, layer, :], s_val)
        metrics.append(
            {
                "layer": layer,
                "validation_r2_log_seconds": m["r2_log_seconds"],
                "validation_corr_log_seconds": m["corr_log_seconds"],
                "validation_long_pause_accuracy": m["long_pause_accuracy"],
            }
        )
    best = max(metrics, key=lambda row: (row["validation_r2_log_seconds"], row["validation_long_pause_accuracy"]))
    return int(best["layer"]), metrics


def evaluate_time_prediction(explicit_features: dict[str, np.ndarray], splits: dict[str, list[dict]], layer: int) -> dict:
    out = {}
    for split in ["train", "validation", "test"]:
        out[split] = time_metrics_for_layer(
            explicit_features["train"][:, layer, :],
            y_seconds(splits["train"]),
            explicit_features[split][:, layer, :],
            y_seconds(splits[split]),
        )
    return out


def fit_direction(x: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    reg = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    reg.fit(x, np.log1p(seconds))
    scaler = reg.named_steps["scale"]
    ridge = reg.named_steps["ridge"]
    coef = ridge.coef_ / scaler.scale_
    return (coef / max(np.linalg.norm(coef), 1e-12)).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


class TimeAdapter(nn.Module):
    def __init__(self, hidden_size: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, hidden_size))

    def forward(self, seconds: torch.Tensor) -> torch.Tensor:
        return self.net(torch.log1p(seconds).view(-1, 1))


def adapter_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    mse = float(np.mean((pred - true) ** 2))
    zero = float(np.mean(true**2))
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    cosines = np.divide(np.sum(pred * true, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
    return {
        "mse": mse,
        "zero_baseline_mse": zero,
        "mse_ratio_vs_zero": float(mse / max(zero, 1e-12)),
        "mean_cosine": float(np.mean(cosines)),
        "median_cosine": float(np.median(cosines)),
    }


def train_adapter_once(train_s, train_delta, val_s, val_delta, test_s, test_delta, width: int, lr: float, epochs: int):
    set_seed(SEED + width + int(lr * 100000))
    adapter = TimeAdapter(train_delta.shape[1], width)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    x_train = torch.tensor(train_s, dtype=torch.float32)
    y_train = torch.tensor(train_delta, dtype=torch.float32)
    x_val = torch.tensor(val_s, dtype=torch.float32)
    y_val = torch.tensor(val_delta, dtype=torch.float32)
    x_test = torch.tensor(test_s, dtype=torch.float32)
    y_test = torch.tensor(test_delta, dtype=torch.float32)
    losses = []
    best_state = None
    best_val = math.inf
    for epoch in range(epochs):
        adapter.train()
        opt.zero_grad()
        pred = adapter(x_train)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            adapter.eval()
            with torch.no_grad():
                val_pred = adapter(x_val)
                test_pred = adapter(x_test)
                val_loss = torch.mean((val_pred - y_val) ** 2)
                test_loss = torch.mean((test_pred - y_test) ** 2)
            row = {"epoch": epoch, "train_mse": float(loss.detach()), "validation_mse": float(val_loss.detach()), "test_mse": float(test_loss.detach())}
            losses.append(row)
            if row["validation_mse"] < best_val:
                best_val = row["validation_mse"]
                best_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    adapter.load_state_dict(best_state)
    adapter.eval()
    with torch.no_grad():
        val_pred_np = adapter(x_val).detach().cpu().numpy()
    return adapter, losses, adapter_metrics(val_pred_np, val_delta)


def train_adapter_grid(deltas: dict[str, np.ndarray], splits: dict[str, list[dict]], epochs: int):
    runs = []
    for width in [128, 256]:
        for lr in [3e-3, 1e-3]:
            adapter, losses, val = train_adapter_once(
                y_seconds(splits["train"]),
                deltas["train"],
                y_seconds(splits["validation"]),
                deltas["validation"],
                y_seconds(splits["test"]),
                deltas["test"],
                width,
                lr,
                epochs,
            )
            runs.append({"width": width, "lr": lr, "adapter": adapter, "losses": losses, "validation": val})
    best = min(runs, key=lambda row: row["validation"]["mse"])
    summary = {
        "selected_width": best["width"],
        "selected_lr": best["lr"],
        "grid": [{k: v for k, v in row.items() if k != "adapter"} for row in runs],
        "selected_losses": best["losses"],
    }
    return best["adapter"], summary


def adapter_predict(adapter: TimeAdapter, seconds: np.ndarray) -> np.ndarray:
    adapter.eval()
    with torch.no_grad():
        return adapter(torch.tensor(seconds, dtype=torch.float32)).detach().cpu().numpy().astype(np.float32)


def make_decision_features(context: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return np.concatenate([context, delta], axis=1).astype(np.float32)


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))).tolist() if False else list(range(len(LABELS)))).tolist(),
        "labels": LABELS,
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        },
        "pred_counts": dict(Counter(LABELS[int(i)] for i in y_pred)),
    }
    if probs is not None:
        out["mean_probabilities"] = {LABELS[i]: float(np.mean(probs[:, i])) for i in range(len(LABELS))}
    return out


class LinearHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Linear(dim, len(LABELS))

    def forward(self, x):
        return self.net(x)


class MLPHead(nn.Module):
    def __init__(self, dim: int, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, width), nn.ReLU(), nn.Dropout(0.1), nn.Linear(width, len(LABELS)))

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainedHead:
    name: str
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    curves: list[dict]
    selected_epoch: int
    selection_score: float


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def predict_head(head: TrainedHead, x: np.ndarray, device: torch.device, batch_size: int = 512):
    head.model.eval()
    x_std = standardize(x, head.mean, head.std)
    probs = []
    with torch.no_grad():
        for start in range(0, len(x_std), batch_size):
            xb = torch.tensor(x_std[start : start + batch_size], dtype=torch.float32, device=device)
            logits = head.model(xb)
            probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    probs_np = np.concatenate(probs, axis=0)
    return np.argmax(probs_np, axis=1), probs_np


def metric_at(head: TrainedHead, features: dict[str, dict[str, np.ndarray]], splits: dict[str, list[dict]], split: str, mode: str, device: torch.device):
    pred, probs = predict_head(head, features[mode][split], device)
    return class_metrics(y_labels(splits[split]), pred, probs)


def train_head(
    name: str,
    architecture: str,
    target: str,
    features: dict[str, dict[str, np.ndarray]],
    splits: dict[str, list[dict]],
    epochs: int,
    device: torch.device,
):
    x_train_raw = np.vstack([features["explicit"]["train"], features["adapter"]["train"]])
    mean, std = standardize_fit(x_train_raw)
    x_train = standardize(x_train_raw, mean, std)
    y_train_hard = np.concatenate([y_labels(splits["train"]), y_labels(splits["train"])])
    y_train_soft = np.vstack([y_soft(splits["train"]), y_soft(splits["train"])])

    model = LinearHead(x_train.shape[1]) if architecture == "linear" else MLPHead(x_train.shape[1], 256)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4 if architecture == "mlp" else 2e-3, weight_decay=1e-4)
    counts = np.bincount(y_train_hard, minlength=len(LABELS)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    hard_tensor = torch.tensor(y_train_hard, dtype=torch.long)
    soft_tensor = torch.tensor(y_train_soft, dtype=torch.float32)
    batch_size = 256
    curves = []
    best_state = None
    best_score = -1.0
    best_epoch = 0
    rng = np.random.default_rng(SEED)

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x_tensor))
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_tensor[idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            if target == "soft":
                yb = soft_tensor[idx].to(device)
                loss = -(yb * torch.log_softmax(logits, dim=-1)).sum(dim=1).mean()
            else:
                yb = hard_tensor[idx].to(device)
                loss = nn.functional.cross_entropy(logits, yb, weight=class_weights)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(idx)

        if epoch % 2 == 0 or epoch == epochs - 1:
            temp_head = TrainedHead(name, model, mean, std, [], epoch, 0.0)
            val_adapter = metric_at(temp_head, features, splits, "validation", "adapter", device)
            val_explicit = metric_at(temp_head, features, splits, "validation", "explicit", device)
            test_adapter = metric_at(temp_head, features, splits, "test", "adapter", device)
            test_explicit = metric_at(temp_head, features, splits, "test", "explicit", device)
            score = (val_adapter["macro_f1"] + val_explicit["macro_f1"]) / 2.0
            row = {
                "epoch": epoch,
                "train_loss": epoch_loss / len(x_tensor),
                "validation_adapter_accuracy": val_adapter["accuracy"],
                "validation_adapter_macro_f1": val_adapter["macro_f1"],
                "validation_explicit_accuracy": val_explicit["accuracy"],
                "validation_explicit_macro_f1": val_explicit["macro_f1"],
                "test_adapter_accuracy": test_adapter["accuracy"],
                "test_adapter_macro_f1": test_adapter["macro_f1"],
                "test_explicit_accuracy": test_explicit["accuracy"],
                "test_explicit_macro_f1": test_explicit["macro_f1"],
            }
            curves.append(row)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return TrainedHead(name, model, mean, std, curves, best_epoch, best_score)


def evaluate_head_all_modes(head: TrainedHead, features: dict[str, dict[str, np.ndarray]], splits: dict[str, list[dict]], device: torch.device) -> dict:
    out = {}
    for mode in ["base", "explicit", "adapter"]:
        out[mode] = {}
        for split in ["train", "validation", "test"]:
            out[mode][split] = metric_at(head, features, splits, split, mode, device)
    explicit_pred, _ = predict_head(head, features["explicit"]["test"], device)
    adapter_pred, _ = predict_head(head, features["adapter"]["test"], device)
    out["explicit_adapter_test_agreement"] = float(np.mean(explicit_pred == adapter_pred))
    return out


def metrics_by_time(rows: list[dict], y_pred: np.ndarray, probs: np.ndarray) -> dict:
    out = {}
    for sec in sorted({row["seconds"] for row in rows}):
        idx = np.array([i for i, row in enumerate(rows) if row["seconds"] == sec], dtype=np.int64)
        out[str(sec)] = class_metrics(y_labels([rows[i] for i in idx]), y_pred[idx], probs[idx])
    return out


def metrics_for_unseen_seconds(rows: list[dict], y_pred: np.ndarray, probs: np.ndarray, train_times: list[float]) -> dict:
    train_set = set(train_times)
    idx = np.array([i for i, row in enumerate(rows) if row["seconds"] not in train_set], dtype=np.int64)
    if len(idx) == 0:
        return {}
    return class_metrics(y_labels([rows[i] for i in idx]), y_pred[idx], probs[idx])


def context_grid(rows: list[dict], pred: np.ndarray, probs: np.ndarray) -> list[dict]:
    grid = []
    for row, pred_id, p in zip(rows, pred, probs):
        grid.append(
            {
                "id": row["id"],
                "context_group_id": row["context_group_id"],
                "profile": row["profile"],
                "seconds": row["seconds"],
                "gold": row["label"],
                "prediction": LABELS[int(pred_id)],
                "fragment": row["fragment"],
                "support_minus_wait": float(p[LABEL_TO_ID["SUPPORT"]] - p[LABEL_TO_ID["WAIT"]]),
                "backchannel_minus_wait": float(p[LABEL_TO_ID["BACKCHANNEL"]] - p[LABEL_TO_ID["WAIT"]]),
                "probabilities": {LABELS[i]: float(p[i]) for i in range(len(LABELS))},
            }
        )
    return grid


def failure_examples(rows: list[dict], pred: np.ndarray, probs: np.ndarray, limit: int = 30) -> list[dict]:
    failures = []
    for row, pred_id, p in zip(rows, pred, probs):
        gold_id = LABEL_TO_ID[row["label"]]
        if int(pred_id) != gold_id:
            failures.append(
                {
                    "id": row["id"],
                    "profile": row["profile"],
                    "context_group_id": row["context_group_id"],
                    "seconds": row["seconds"],
                    "time_expression": row["time_expression"],
                    "fragment": row["fragment"],
                    "gold": row["label"],
                    "prediction": LABELS[int(pred_id)],
                    "acceptable_labels": row["acceptable_labels"],
                    "soft_label": row["soft_label"],
                    "rationale": row["rationale"],
                    "probabilities": {LABELS[i]: float(p[i]) for i in range(len(LABELS))},
                    "confidence": float(np.max(p)),
                }
            )
    failures.sort(key=lambda r: r["confidence"], reverse=True)
    return failures[:limit]


def non_time_probe_rows(rows: list[dict], count: int = 32) -> list[dict]:
    selected = []
    seen = Counter()
    profiles = sorted({r["profile"] for r in rows})
    per_profile = max(1, count // len(profiles))
    for row in rows:
        if seen[row["profile"]] < per_profile:
            selected.append(row)
            seen[row["profile"]] += 1
        if len(selected) >= count:
            break
    return selected[:count]


def run_non_time_controls(tokenizer, model, device, head: TrainedHead, context_test: np.ndarray, adapter_vectors_test: np.ndarray, time_direction: np.ndarray, layer: int, train_rows: list[dict], test_rows: list[dict], batch_size: int):
    units = {"kg": "kg", "m": "m", "score": "点", "yen": "円"}
    values = [0.5, 1.0, 2.0, 5.0, 8.0]
    probes = non_time_probe_rows(train_rows, 32)
    controls = {}
    norms = np.linalg.norm(adapter_vectors_test, axis=1).reshape(-1, 1)
    for name, unit in units.items():
        prompts, numeric_values = [], []
        for row in probes:
            for value in values:
                prompts.append(build_prompt(row, include_timing=False, override_control=f"[{value:g}{unit}]"))
                numeric_values.append(value)
        hidden = extract_hidden(tokenizer, model, device, prompts, out_path=None, batch_size=batch_size)
        direction = fit_direction(hidden[:, layer, :], np.array(numeric_values, dtype=np.float32))
        delta = direction.reshape(1, -1) * norms
        features = make_decision_features(context_test, delta)
        pred, probs = predict_head(head, features, device)
        controls[name] = {
            "cosine_to_time_direction": cosine(direction, time_direction),
            "metrics": class_metrics(y_labels(test_rows), pred, probs),
        }
    return controls


def run_random_baseline(head: TrainedHead, context_test: np.ndarray, adapter_vectors_test: np.ndarray, test_rows: list[dict], device: torch.device, trials: int = 5):
    rng = np.random.default_rng(SEED)
    norms = np.linalg.norm(adapter_vectors_test, axis=1).reshape(-1, 1)
    rows = []
    for trial in range(trials):
        direction = rng.normal(size=context_test.shape[1]).astype(np.float32)
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        delta = direction.reshape(1, -1) * norms
        features = make_decision_features(context_test, delta)
        pred, probs = predict_head(head, features, device)
        rows.append({"trial": trial, "metrics": class_metrics(y_labels(test_rows), pred, probs)})
    return {
        "trials": rows,
        "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in rows])),
        "mean_macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in rows])),
    }


def plot_time_layers(metrics: list[dict], path: Path):
    layers = [row["layer"] for row in metrics]
    plt.figure(figsize=(8, 4.5))
    plt.plot(layers, [row["validation_r2_log_seconds"] for row in metrics], label="R2", marker="o")
    plt.plot(layers, [row["validation_corr_log_seconds"] for row in metrics], label="Corr", marker="s")
    plt.plot(layers, [row["validation_long_pause_accuracy"] for row in metrics], label="Long acc", marker="^")
    plt.xlabel("Layer")
    plt.ylabel("Validation score")
    plt.title("Qwen3-4B phase1 hidden-state time prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_adapter_losses(losses: list[dict], path: Path):
    plt.figure(figsize=(7, 4.2))
    plt.plot([r["epoch"] for r in losses], [r["train_mse"] for r in losses], label="Train")
    plt.plot([r["epoch"] for r in losses], [r["validation_mse"] for r in losses], label="Validation")
    plt.plot([r["epoch"] for r in losses], [r["test_mse"] for r in losses], label="Test")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Time Adapter MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_head_curves(heads: dict[str, TrainedHead], path: Path):
    plt.figure(figsize=(8, 4.8))
    for name, head in heads.items():
        epochs = [r["epoch"] for r in head.curves]
        plt.plot(epochs, [r["validation_adapter_macro_f1"] for r in head.curves], label=f"{name} val")
        plt.plot(epochs, [r["test_adapter_macro_f1"] for r in head.curves], linestyle="--", label=f"{name} test")
    plt.xlabel("Epoch")
    plt.ylabel("Adapter macro F1")
    plt.title("Decision head learning curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_confusion(cm: list[list[int]], path: Path, title: str):
    arr = np.array(cm)
    plt.figure(figsize=(4.7, 4.2))
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


def plot_context_heatmap(grid: list[dict], key: str, path: Path, title: str):
    profiles = sorted({row["profile"] for row in grid})
    seconds = sorted({row["seconds"] for row in grid})
    data = np.full((len(profiles), len(seconds)), np.nan)
    for i, profile in enumerate(profiles):
        for j, sec in enumerate(seconds):
            vals = [row[key] for row in grid if row["profile"] == profile and row["seconds"] == sec]
            if vals:
                data[i, j] = float(np.mean(vals))
    plt.figure(figsize=(8.4, 4.8))
    im = plt.imshow(data, aspect="auto", cmap="PiYG", vmin=-1, vmax=1)
    plt.colorbar(im, label=key)
    plt.xticks(range(len(seconds)), [str(s) for s in seconds])
    plt.yticks(range(len(profiles)), profiles)
    plt.xlabel("Seconds")
    plt.title(title)
    for i in range(len(profiles)):
        for j in range(len(seconds)):
            if not np.isnan(data[i, j]):
                plt.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_baselines(selected_metrics: dict, random_summary: dict, non_time: dict, path: Path):
    names = ["no-time", "explicit", "adapter", "random"] + [f"non-time {k}" for k in non_time]
    accs = [
        selected_metrics["base"]["test"]["accuracy"],
        selected_metrics["explicit"]["test"]["accuracy"],
        selected_metrics["adapter"]["test"]["accuracy"],
        random_summary["mean_accuracy"],
    ] + [non_time[k]["metrics"]["accuracy"] for k in non_time]
    f1s = [
        selected_metrics["base"]["test"]["macro_f1"],
        selected_metrics["explicit"]["test"]["macro_f1"],
        selected_metrics["adapter"]["test"]["macro_f1"],
        random_summary["mean_macro_f1"],
    ] + [non_time[k]["metrics"]["macro_f1"] for k in non_time]
    x = np.arange(len(names))
    plt.figure(figsize=(9, 4.4))
    plt.bar(x - 0.18, accs, width=0.36, label="Accuracy")
    plt.bar(x + 0.18, f1s, width=0.36, label="Macro F1")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Phase1 baseline comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--adapter-epochs", type=int, default=260)
    parser.add_argument("--head-epochs", type=int, default=80)
    args = parser.parse_args()

    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hidden_dir = OUT_DIR / "hidden_cache"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    splits, manifest = load_dataset()
    tokenizer, model, qwen_device = load_model(CACHE_DIR)
    print(f"Loaded {MODEL_ID} on {qwen_device}", flush=True)

    explicit_hidden, base_hidden = {}, {}
    for split, rows in splits.items():
        explicit_hidden[split] = extract_hidden(
            tokenizer,
            model,
            qwen_device,
            [build_prompt(row, include_timing=True) for row in rows],
            hidden_dir / f"explicit_{split}.npy",
            args.batch_size,
        )
        print(f"Explicit hidden {split}: {explicit_hidden[split].shape}", flush=True)
        base_hidden[split] = extract_hidden(
            tokenizer,
            model,
            qwen_device,
            [build_prompt(row, include_timing=False) for row in rows],
            hidden_dir / f"base_{split}.npy",
            args.batch_size,
        )
        print(f"Base hidden {split}: {base_hidden[split].shape}", flush=True)

    layer, layer_metrics = select_time_layer(explicit_hidden, splits)
    time_prediction = evaluate_time_prediction(explicit_hidden, splits, layer)
    print(f"Selected layer {layer}", flush=True)

    explicit_features = {split: explicit_hidden[split][:, layer, :] for split in splits}
    base_features = {split: base_hidden[split][:, layer, :] for split in splits}
    deltas = {split: explicit_features[split] - base_features[split] for split in splits}
    time_direction = fit_direction(explicit_features["train"], y_seconds(splits["train"]))
    np.save(OUT_DIR / "time_direction.npy", time_direction)

    adapter, adapter_summary = train_adapter_grid(deltas, splits, args.adapter_epochs)
    torch.save({"state_dict": adapter.state_dict(), "width": adapter_summary["selected_width"], "layer": layer, "model_id": MODEL_ID}, OUT_DIR / "time_adapter_phase1.pt")
    adapter_vectors = {split: adapter_predict(adapter, y_seconds(splits[split])) for split in splits}
    zero_vectors = {split: np.zeros_like(adapter_vectors[split]) for split in splits}
    adapter_eval = {split: adapter_metrics(adapter_vectors[split], deltas[split]) for split in splits}

    features = {
        "base": {split: make_decision_features(base_features[split], zero_vectors[split]) for split in splits},
        "explicit": {split: make_decision_features(base_features[split], deltas[split]) for split in splits},
        "adapter": {split: make_decision_features(base_features[split], adapter_vectors[split]) for split in splits},
    }

    head_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = {
        "hard_linear": train_head("hard_linear", "linear", "hard", features, splits, args.head_epochs, head_device),
        "hard_mlp": train_head("hard_mlp", "mlp", "hard", features, splits, args.head_epochs, head_device),
        "soft_mlp": train_head("soft_mlp", "mlp", "soft", features, splits, args.head_epochs, head_device),
    }
    head_metrics = {name: evaluate_head_all_modes(head, features, splits, head_device) for name, head in heads.items()}
    selected_name = max(head_metrics, key=lambda n: (head_metrics[n]["adapter"]["validation"]["macro_f1"] + head_metrics[n]["explicit"]["validation"]["macro_f1"]) / 2.0)
    selected_head = heads[selected_name]
    with (OUT_DIR / "decision_heads_phase1.pkl").open("wb") as f:
        pickle.dump(
            {
                name: {
                    "state_dict": head.model.state_dict(),
                    "mean": head.mean,
                    "std": head.std,
                    "selected_epoch": head.selected_epoch,
                    "selection_score": head.selection_score,
                }
                for name, head in heads.items()
            },
            f,
        )

    adapter_pred, adapter_probs = predict_head(selected_head, features["adapter"]["test"], head_device)
    explicit_pred, explicit_probs = predict_head(selected_head, features["explicit"]["test"], head_device)
    base_pred, base_probs = predict_head(selected_head, features["base"]["test"], head_device)

    random_summary = run_random_baseline(selected_head, base_features["test"], adapter_vectors["test"], splits["test"], head_device)
    non_time = run_non_time_controls(
        tokenizer,
        model,
        qwen_device,
        selected_head,
        base_features["test"],
        adapter_vectors["test"],
        time_direction,
        layer,
        splits["train"],
        splits["test"],
        args.batch_size,
    )

    selected_metrics = head_metrics[selected_name]
    test_rows = splits["test"]
    adapter_grid = context_grid(test_rows, adapter_pred, adapter_probs)
    explicit_grid = context_grid(test_rows, explicit_pred, explicit_probs)
    failure_sets = {
        "adapter": failure_examples(test_rows, adapter_pred, adapter_probs),
        "explicit": failure_examples(test_rows, explicit_pred, explicit_probs),
        "base": failure_examples(test_rows, base_pred, base_probs),
    }

    figure_paths = {
        "time_layer_metrics": str(FIG_DIR / "time_layer_metrics.png"),
        "adapter_losses": str(FIG_DIR / "adapter_losses.png"),
        "head_curves": str(FIG_DIR / "head_curves.png"),
        "confusion_adapter_test": str(FIG_DIR / "confusion_adapter_test.png"),
        "confusion_explicit_test": str(FIG_DIR / "confusion_explicit_test.png"),
        "confusion_base_test": str(FIG_DIR / "confusion_base_test.png"),
        "support_wait_heatmap_adapter": str(FIG_DIR / "support_wait_heatmap_adapter.png"),
        "backchannel_wait_heatmap_adapter": str(FIG_DIR / "backchannel_wait_heatmap_adapter.png"),
        "support_wait_heatmap_explicit": str(FIG_DIR / "support_wait_heatmap_explicit.png"),
        "baseline_comparison": str(FIG_DIR / "baseline_comparison.png"),
    }
    plot_time_layers(layer_metrics, Path(figure_paths["time_layer_metrics"]))
    plot_adapter_losses(adapter_summary["selected_losses"], Path(figure_paths["adapter_losses"]))
    plot_head_curves(heads, Path(figure_paths["head_curves"]))
    plot_confusion(selected_metrics["adapter"]["test"]["confusion_matrix"], Path(figure_paths["confusion_adapter_test"]), f"{selected_name}: adapter test")
    plot_confusion(selected_metrics["explicit"]["test"]["confusion_matrix"], Path(figure_paths["confusion_explicit_test"]), f"{selected_name}: explicit test")
    plot_confusion(selected_metrics["base"]["test"]["confusion_matrix"], Path(figure_paths["confusion_base_test"]), f"{selected_name}: no-time test")
    plot_context_heatmap(adapter_grid, "support_minus_wait", Path(figure_paths["support_wait_heatmap_adapter"]), "Adapter SUPPORT-WAIT")
    plot_context_heatmap(adapter_grid, "backchannel_minus_wait", Path(figure_paths["backchannel_wait_heatmap_adapter"]), "Adapter BACKCHANNEL-WAIT")
    plot_context_heatmap(explicit_grid, "support_minus_wait", Path(figure_paths["support_wait_heatmap_explicit"]), "Explicit SUPPORT-WAIT")
    plot_baselines(selected_metrics, random_summary, non_time, Path(figure_paths["baseline_comparison"]))

    summary = {
        "model_id": MODEL_ID,
        "device": str(qwen_device),
        "head_device": str(head_device),
        "dataset_manifest": manifest,
        "num_layers": int(explicit_hidden["train"].shape[1]),
        "hidden_size": int(explicit_hidden["train"].shape[2]),
        "selected_layer": layer,
        "layer_selection_metrics": layer_metrics,
        "time_prediction": time_prediction,
        "adapter": {**adapter_summary, "evaluation": adapter_eval},
        "heads": {
            name: {
                "selected_epoch": head.selected_epoch,
                "selection_score": head.selection_score,
                "curves": head.curves,
                "metrics": head_metrics[name],
            }
            for name, head in heads.items()
        },
        "selected_head": selected_name,
        "selected_head_metrics": selected_metrics,
        "explicit_adapter_test_agreement": selected_metrics["explicit_adapter_test_agreement"],
        "metrics_by_time": {
            "adapter_test": metrics_by_time(test_rows, adapter_pred, adapter_probs),
            "explicit_test": metrics_by_time(test_rows, explicit_pred, explicit_probs),
        },
        "unseen_seconds_metrics": {
            "adapter_test": metrics_for_unseen_seconds(test_rows, adapter_pred, adapter_probs, manifest["train_times"]),
            "explicit_test": metrics_for_unseen_seconds(test_rows, explicit_pred, explicit_probs, manifest["train_times"]),
        },
        "unseen_utterance_metrics": {
            "test_all_utterances_unseen_by_split": selected_metrics["adapter"]["test"],
        },
        "random_vector_baseline": random_summary,
        "non_time_controls": non_time,
        "context_x_time": {"adapter_grid": adapter_grid, "explicit_grid": explicit_grid},
        "failures": failure_sets,
        "figures": figure_paths,
        "artifacts": {
            "hidden_cache": str(hidden_dir),
            "time_direction": str(OUT_DIR / "time_direction.npy"),
            "time_adapter": str(OUT_DIR / "time_adapter_phase1.pt"),
            "decision_heads": str(OUT_DIR / "decision_heads_phase1.pkl"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
