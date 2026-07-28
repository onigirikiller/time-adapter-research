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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATA_DIR = ROOT / "data/qwen3_context_time_expanded"
OUT_DIR = ROOT / "artifacts/qwen3_expanded_training"
FIG_DIR = ROOT / "output/figures/qwen3_expanded_training"
CACHE_DIR = ROOT / ".cache/huggingface"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
SEED = 20260622


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(cache_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=str(cache_dir),
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=str(cache_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def extract_hidden(
    tokenizer,
    model,
    device: torch.device,
    prompts: list[str],
    out_path: Path | None = None,
    batch_size: int = 8,
) -> np.ndarray:
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
            batch_layers = []
            batch_index = torch.arange(len(batch), device=device)
            for hidden in outputs.hidden_states[1:]:
                selected = hidden[batch_index, lengths, :].detach().float().cpu().numpy().astype(np.float32)
                batch_layers.append(selected)
            rows.append(np.stack(batch_layers, axis=1))
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


def fit_time_models_for_layer(x_train: np.ndarray, s_train: np.ndarray, x_eval: np.ndarray, s_eval: np.ndarray):
    y_train = np.log1p(s_train)
    y_eval = np.log1p(s_eval)
    reg = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    reg.fit(x_train, y_train)
    pred = reg.predict(x_eval)
    corr = float(np.corrcoef(y_eval, pred)[0, 1]) if len(set(np.round(y_eval, 6))) > 1 else 0.0

    cls_train = (s_train >= 2.0).astype(np.int64)
    cls_eval = (s_eval >= 2.0).astype(np.int64)
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(C=0.5, max_iter=1000, solver="liblinear")),
        ]
    )
    clf.fit(x_train, cls_train)
    cls_pred = clf.predict(x_eval)
    return {
        "r2_log_seconds": float(r2_score(y_eval, pred)),
        "corr_log_seconds": corr,
        "long_pause_accuracy": float(accuracy_score(cls_eval, cls_pred)),
        "regressor": reg,
        "classifier": clf,
    }


def select_time_layer(explicit_hidden: dict[str, np.ndarray], splits: dict[str, list[dict]]) -> tuple[int, list[dict]]:
    s_train = y_seconds(splits["train"])
    s_val = y_seconds(splits["validation"])
    metrics = []
    n_layers = explicit_hidden["train"].shape[1]
    for layer in range(n_layers):
        result = fit_time_models_for_layer(
            explicit_hidden["train"][:, layer, :],
            s_train,
            explicit_hidden["validation"][:, layer, :],
            s_val,
        )
        metrics.append(
            {
                "layer": layer,
                "validation_r2_log_seconds": result["r2_log_seconds"],
                "validation_corr_log_seconds": result["corr_log_seconds"],
                "validation_long_pause_accuracy": result["long_pause_accuracy"],
            }
        )
    best = max(metrics, key=lambda m: (m["validation_r2_log_seconds"], m["validation_long_pause_accuracy"]))
    return int(best["layer"]), metrics


def evaluate_time_prediction(
    explicit_hidden: dict[str, np.ndarray],
    splits: dict[str, list[dict]],
    layer: int,
) -> dict:
    s_train = y_seconds(splits["train"])
    out = {}
    for split in ["train", "validation", "test"]:
        result = fit_time_models_for_layer(
            explicit_hidden["train"][:, layer, :],
            s_train,
            explicit_hidden[split][:, layer, :],
            y_seconds(splits[split]),
        )
        out[split] = {
            "r2_log_seconds": result["r2_log_seconds"],
            "corr_log_seconds": result["corr_log_seconds"],
            "long_pause_accuracy": result["long_pause_accuracy"],
        }
    return out


def fit_direction(x: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    reg = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    reg.fit(x, np.log1p(seconds))
    scaler = reg.named_steps["scale"]
    ridge = reg.named_steps["ridge"]
    coef = ridge.coef_ / scaler.scale_
    norm = np.linalg.norm(coef)
    return (coef / max(norm, 1e-12)).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


class TimeAdapter(nn.Module):
    def __init__(self, hidden_size: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, hidden_size))

    def forward(self, seconds: torch.Tensor) -> torch.Tensor:
        return self.net(torch.log1p(seconds).view(-1, 1))


@dataclass
class AdapterRun:
    width: int
    lr: float
    losses: list[dict]
    state_dict: dict
    validation: dict


def adapter_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    mse = float(np.mean((pred - true) ** 2))
    zero_mse = float(np.mean(true**2))
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    cosines = np.divide(np.sum(pred * true, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
    return {
        "mse": mse,
        "zero_baseline_mse": zero_mse,
        "mse_ratio_vs_zero": float(mse / max(zero_mse, 1e-12)),
        "mean_cosine": float(np.mean(cosines)),
        "median_cosine": float(np.median(cosines)),
    }


def train_one_adapter(
    train_seconds: np.ndarray,
    train_delta: np.ndarray,
    val_seconds: np.ndarray,
    val_delta: np.ndarray,
    width: int,
    lr: float,
    epochs: int,
) -> AdapterRun:
    set_seed(SEED + width + int(lr * 100000))
    adapter = TimeAdapter(train_delta.shape[1], width)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    x_train = torch.tensor(train_seconds, dtype=torch.float32)
    y_train = torch.tensor(train_delta, dtype=torch.float32)
    x_val = torch.tensor(val_seconds, dtype=torch.float32)
    y_val = torch.tensor(val_delta, dtype=torch.float32)
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
                val_loss = torch.mean((val_pred - y_val) ** 2)
            row = {"epoch": epoch, "train_mse": float(loss.detach()), "validation_mse": float(val_loss.detach())}
            losses.append(row)
            if row["validation_mse"] < best_val:
                best_val = row["validation_mse"]
                best_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    adapter.load_state_dict(best_state)
    adapter.eval()
    with torch.no_grad():
        val_pred_np = adapter(x_val).detach().cpu().numpy()
    return AdapterRun(
        width=width,
        lr=lr,
        losses=losses,
        state_dict=best_state,
        validation=adapter_metrics(val_pred_np, val_delta),
    )


def train_adapter_grid(deltas: dict[str, np.ndarray], splits: dict[str, list[dict]], epochs: int) -> tuple[TimeAdapter, dict]:
    train_s = y_seconds(splits["train"])
    val_s = y_seconds(splits["validation"])
    runs = []
    for width in [64, 128]:
        for lr in [3e-3, 1e-3]:
            runs.append(train_one_adapter(train_s, deltas["train"], val_s, deltas["validation"], width, lr, epochs))
    best = min(runs, key=lambda r: r.validation["mse"])
    adapter = TimeAdapter(deltas["train"].shape[1], best.width)
    adapter.load_state_dict(best.state_dict)
    adapter.eval()
    summary = {
        "selected_width": best.width,
        "selected_lr": best.lr,
        "grid": [
            {
                "width": run.width,
                "lr": run.lr,
                "validation": run.validation,
                "losses": run.losses,
            }
            for run in runs
        ],
        "selected_losses": best.losses,
    }
    return adapter, summary


def adapter_predict(adapter: TimeAdapter, seconds: np.ndarray) -> np.ndarray:
    adapter.eval()
    with torch.no_grad():
        return adapter(torch.tensor(seconds, dtype=torch.float32)).detach().cpu().numpy().astype(np.float32)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
        "labels": LABELS,
    }


def make_decision_features(context_features: np.ndarray, time_delta_features: np.ndarray) -> np.ndarray:
    return np.concatenate([context_features, time_delta_features], axis=1)


def train_decision_head(features: dict[str, dict[str, np.ndarray]], splits: dict[str, list[dict]]) -> tuple[Pipeline, dict]:
    y_train = y_labels(splits["train"])
    y_val = y_labels(splits["validation"])
    x_train = np.vstack([features["explicit"]["train"], features["adapter"]["train"]])
    yy_train = np.concatenate([y_train, y_train])
    candidates = []
    selected = None
    rows = []
    for c in [0.05, 0.1, 0.5, 1.0, 2.0]:
        candidates.append(
            (
                f"linear_logreg_C{c}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "logreg",
                            LogisticRegression(
                                C=c,
                                max_iter=1500,
                                solver="lbfgs",
                                class_weight="balanced",
                            ),
                        ),
                    ]
                ),
            )
        )
    for hidden, alpha in [((128,), 1e-4), ((256,), 1e-4), ((256, 64), 5e-4)]:
        candidates.append(
            (
                f"mlp_{'-'.join(map(str, hidden))}_alpha{alpha:g}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "mlp",
                            MLPClassifier(
                                hidden_layer_sizes=hidden,
                                alpha=alpha,
                                learning_rate_init=1e-3,
                                max_iter=500,
                                early_stopping=True,
                                validation_fraction=0.12,
                                n_iter_no_change=20,
                                random_state=SEED,
                            ),
                        ),
                    ]
                ),
            )
        )

    for name, head in candidates:
        head.fit(x_train, yy_train)
        explicit_pred = head.predict(features["explicit"]["validation"])
        adapter_pred = head.predict(features["adapter"]["validation"])
        explicit_m = classification_metrics(y_val, explicit_pred)
        adapter_m = classification_metrics(y_val, adapter_pred)
        score = (explicit_m["macro_f1"] + adapter_m["macro_f1"]) / 2.0
        row = {
            "candidate": name,
            "selection_score": score,
            "validation_explicit": explicit_m,
            "validation_adapter": adapter_m,
        }
        rows.append(row)
        if selected is None or score > selected[0]:
            selected = (score, name, head)
    return selected[2], {"selected_candidate": selected[1], "validation_grid": rows}


def predict_with_prob(head: Pipeline, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return head.predict(x), head.predict_proba(x)


def summarize_predictions(head: Pipeline, features: dict[str, dict[str, np.ndarray]], splits: dict[str, list[dict]]) -> dict:
    out = {}
    for mode in ["explicit", "adapter", "base"]:
        out[mode] = {}
        for split in ["train", "validation", "test"]:
            pred, prob = predict_with_prob(head, features[mode][split])
            out[mode][split] = {
                **classification_metrics(y_labels(splits[split]), pred),
                "pred_counts": dict(Counter(LABELS[int(i)] for i in pred)),
                "mean_probabilities": {LABELS[i]: float(np.mean(prob[:, i])) for i in range(len(LABELS))},
            }
    explicit_test, _ = predict_with_prob(head, features["explicit"]["test"])
    adapter_test, _ = predict_with_prob(head, features["adapter"]["test"])
    out["explicit_adapter_test_agreement"] = float(np.mean(explicit_test == adapter_test))
    return out


def non_time_probe_rows(rows: list[dict], count: int = 24) -> list[dict]:
    selected = []
    seen_profiles = Counter()
    for row in rows:
        if seen_profiles[row["profile"]] < max(1, count // len(set(r["profile"] for r in rows))):
            selected.append(row)
            seen_profiles[row["profile"]] += 1
        if len(selected) >= count:
            break
    if len(selected) < count:
        selected.extend(rows[: count - len(selected)])
    return selected[:count]


def run_non_time_controls(
    tokenizer,
    model,
    device,
    head: Pipeline,
    context_features_test: np.ndarray,
    adapter_vectors_test: np.ndarray,
    time_direction: np.ndarray,
    layer: int,
    train_rows: list[dict],
    test_labels: np.ndarray,
    batch_size: int,
) -> dict:
    units = {"kg": "kg", "m": "m", "score": "点", "yen": "円"}
    values = [0.5, 1.0, 2.0, 5.0, 8.0]
    probes = non_time_probe_rows(train_rows, count=24)
    controls = {}
    for name, unit in units.items():
        prompts = []
        numeric_values = []
        for row in probes:
            for value in values:
                prompts.append(build_prompt(row, include_timing=False, override_control=f"[{value:g}{unit}]"))
                numeric_values.append(value)
        hidden = extract_hidden(tokenizer, model, device, prompts, out_path=None, batch_size=batch_size)
        x = hidden[:, layer, :]
        direction = fit_direction(x, np.array(numeric_values, dtype=np.float32))
        scaled_delta = direction.reshape(1, -1) * np.linalg.norm(adapter_vectors_test, axis=1).reshape(-1, 1)
        scaled = make_decision_features(context_features_test, scaled_delta)
        pred = head.predict(scaled)
        controls[name] = {
            "cosine_to_time_direction": cosine(direction, time_direction),
            "intervention_metrics": classification_metrics(test_labels, pred),
            "pred_counts": dict(Counter(LABELS[int(i)] for i in pred)),
        }
    return controls


def run_random_baseline(
    head: Pipeline,
    context_features_test: np.ndarray,
    adapter_vectors_test: np.ndarray,
    test_labels: np.ndarray,
    trials: int = 5,
) -> dict:
    rng = np.random.default_rng(SEED)
    rows = []
    norms = np.linalg.norm(adapter_vectors_test, axis=1).reshape(-1, 1)
    for trial in range(trials):
        direction = rng.normal(size=context_features_test.shape[1]).astype(np.float32)
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        random_delta = direction.reshape(1, -1) * norms
        x = make_decision_features(context_features_test, random_delta)
        pred = head.predict(x)
        rows.append(
            {
                "trial": trial,
                "metrics": classification_metrics(test_labels, pred),
                "pred_counts": dict(Counter(LABELS[int(i)] for i in pred)),
            }
        )
    return {
        "trials": rows,
        "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in rows])),
        "mean_macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in rows])),
    }


def context_grid(rows: list[dict], pred: np.ndarray, prob: np.ndarray) -> list[dict]:
    out = []
    for row, pred_id, p in zip(rows, pred, prob):
        out.append(
            {
                "id": row["id"],
                "profile": row["profile"],
                "seconds": row["seconds"],
                "gold": row["label"],
                "prediction": LABELS[int(pred_id)],
                "fragment": row["fragment"],
                "support_minus_wait_probability": float(p[LABEL_TO_ID["SUPPORT"]] - p[LABEL_TO_ID["WAIT"]]),
                "probabilities": {LABELS[i]: float(p[i]) for i in range(len(LABELS))},
            }
        )
    return out


def failure_examples(rows: list[dict], pred: np.ndarray, prob: np.ndarray, limit: int = 20) -> list[dict]:
    failures = []
    for row, pred_id, p in zip(rows, pred, prob):
        gold_id = LABEL_TO_ID[row["label"]]
        if int(pred_id) != gold_id:
            failures.append(
                {
                    "id": row["id"],
                    "profile": row["profile"],
                    "seconds": row["seconds"],
                    "time_expression": row["time_expression"],
                    "fragment": row["fragment"],
                    "gold": row["label"],
                    "prediction": LABELS[int(pred_id)],
                    "acceptable_labels": row["acceptable_labels"],
                    "rationale": row["rationale"],
                    "probabilities": {LABELS[i]: float(p[i]) for i in range(len(LABELS))},
                }
            )
    failures.sort(key=lambda r: r["probabilities"][r["prediction"]], reverse=True)
    return failures[:limit]


def plot_time_layers(metrics: list[dict], path: Path):
    layers = [row["layer"] for row in metrics]
    r2 = [row["validation_r2_log_seconds"] for row in metrics]
    corr = [row["validation_corr_log_seconds"] for row in metrics]
    acc = [row["validation_long_pause_accuracy"] for row in metrics]
    plt.figure(figsize=(8, 4.6))
    plt.plot(layers, r2, marker="o", label="R2")
    plt.plot(layers, corr, marker="s", label="Correlation")
    plt.plot(layers, acc, marker="^", label="Long-pause acc")
    plt.xlabel("Layer")
    plt.ylabel("Validation score")
    plt.title("Hidden-state time prediction by layer")
    plt.ylim(min(-0.1, min(r2) - 0.05), 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_adapter_losses(losses: list[dict], path: Path):
    epochs = [row["epoch"] for row in losses]
    train = [row["train_mse"] for row in losses]
    val = [row["validation_mse"] for row in losses]
    plt.figure(figsize=(7, 4.4))
    plt.plot(epochs, train, label="Train MSE")
    plt.plot(epochs, val, label="Validation MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Selected Time Adapter training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_confusion(cm: list[list[int]], path: Path, title: str):
    arr = np.array(cm)
    plt.figure(figsize=(4.7, 4.3))
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


def plot_context_heatmap(grid: list[dict], path: Path, title: str):
    profiles = sorted(set(row["profile"] for row in grid))
    seconds = sorted(set(row["seconds"] for row in grid))
    data = np.full((len(profiles), len(seconds)), np.nan)
    for i, profile in enumerate(profiles):
        for j, sec in enumerate(seconds):
            vals = [row["support_minus_wait_probability"] for row in grid if row["profile"] == profile and row["seconds"] == sec]
            if vals:
                data[i, j] = float(np.mean(vals))
    plt.figure(figsize=(8.2, 4.8))
    im = plt.imshow(data, aspect="auto", cmap="PiYG", vmin=-1, vmax=1)
    plt.colorbar(im, label="P(SUPPORT) - P(WAIT)")
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


def plot_non_time(non_time: dict, path: Path):
    names = list(non_time.keys())
    cos = [non_time[name]["cosine_to_time_direction"] for name in names]
    acc = [non_time[name]["intervention_metrics"]["accuracy"] for name in names]
    x = np.arange(len(names))
    plt.figure(figsize=(7, 4.2))
    plt.bar(x - 0.18, cos, width=0.36, label="Cosine to time direction")
    plt.bar(x + 0.18, acc, width=0.36, label="Intervention accuracy")
    plt.xticks(x, names)
    plt.ylim(-0.2, 1.05)
    plt.title("Non-time numeric direction controls")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--adapter-epochs", type=int, default=260)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hidden_dir = OUT_DIR / "hidden_cache"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    splits, manifest = load_dataset()
    tokenizer, model, device = load_model(CACHE_DIR)
    print(f"Loaded {MODEL_ID} on {device}", flush=True)

    explicit_hidden = {}
    base_hidden = {}
    for split, rows in splits.items():
        explicit_prompts = [build_prompt(row, include_timing=True) for row in rows]
        base_prompts = [build_prompt(row, include_timing=False) for row in rows]
        explicit_hidden[split] = extract_hidden(
            tokenizer,
            model,
            device,
            explicit_prompts,
            hidden_dir / f"explicit_{split}.npy",
            batch_size=args.batch_size,
        )
        print(f"Explicit hidden {split}: {explicit_hidden[split].shape}", flush=True)
        base_hidden[split] = extract_hidden(
            tokenizer,
            model,
            device,
            base_prompts,
            hidden_dir / f"base_{split}.npy",
            batch_size=args.batch_size,
        )
        print(f"Base hidden {split}: {base_hidden[split].shape}", flush=True)

    best_layer, layer_metrics = select_time_layer(explicit_hidden, splits)
    time_prediction = evaluate_time_prediction(explicit_hidden, splits, best_layer)
    print(f"Selected layer {best_layer}", flush=True)

    explicit_features = {split: explicit_hidden[split][:, best_layer, :] for split in splits}
    base_features = {split: base_hidden[split][:, best_layer, :] for split in splits}
    deltas = {split: explicit_features[split] - base_features[split] for split in splits}

    time_direction = fit_direction(explicit_features["train"], y_seconds(splits["train"]))
    np.save(OUT_DIR / "time_direction.npy", time_direction)

    adapter, adapter_summary = train_adapter_grid(deltas, splits, epochs=args.adapter_epochs)
    torch.save(
        {"state_dict": adapter.state_dict(), "width": adapter_summary["selected_width"], "model_id": MODEL_ID, "layer": best_layer},
        OUT_DIR / "time_adapter_expanded.pt",
    )
    adapter_vectors = {split: adapter_predict(adapter, y_seconds(splits[split])) for split in splits}
    adapter_features = {split: base_features[split] + adapter_vectors[split] for split in splits}
    adapter_eval = {}
    for split in splits:
        adapter_eval[split] = adapter_metrics(adapter_vectors[split], deltas[split])

    zero_vectors = {split: np.zeros_like(adapter_vectors[split]) for split in splits}
    decision_features = {
        "explicit": {split: make_decision_features(base_features[split], deltas[split]) for split in splits},
        "adapter": {split: make_decision_features(base_features[split], adapter_vectors[split]) for split in splits},
        "base": {split: make_decision_features(base_features[split], zero_vectors[split]) for split in splits},
    }
    head, head_summary = train_decision_head(decision_features, splits)
    with (OUT_DIR / "decision_head.pkl").open("wb") as f:
        pickle.dump(head, f)
    prediction_summary = summarize_predictions(head, decision_features, splits)

    test_labels = y_labels(splits["test"])
    adapter_pred_test, adapter_prob_test = predict_with_prob(head, decision_features["adapter"]["test"])
    explicit_pred_test, explicit_prob_test = predict_with_prob(head, decision_features["explicit"]["test"])
    base_pred_test, base_prob_test = predict_with_prob(head, decision_features["base"]["test"])

    random_summary = run_random_baseline(head, base_features["test"], adapter_vectors["test"], test_labels)
    non_time = run_non_time_controls(
        tokenizer,
        model,
        device,
        head,
        base_features["test"],
        adapter_vectors["test"],
        time_direction,
        best_layer,
        splits["train"],
        test_labels,
        args.batch_size,
    )

    adapter_grid = context_grid(splits["test"], adapter_pred_test, adapter_prob_test)
    explicit_grid = context_grid(splits["test"], explicit_pred_test, explicit_prob_test)
    failures = {
        "adapter": failure_examples(splits["test"], adapter_pred_test, adapter_prob_test),
        "explicit": failure_examples(splits["test"], explicit_pred_test, explicit_prob_test),
        "base": failure_examples(splits["test"], base_pred_test, base_prob_test),
    }

    figure_paths = {
        "time_layer_metrics": str(FIG_DIR / "time_layer_metrics.png"),
        "adapter_losses": str(FIG_DIR / "adapter_losses.png"),
        "confusion_explicit_test": str(FIG_DIR / "confusion_explicit_test.png"),
        "confusion_adapter_test": str(FIG_DIR / "confusion_adapter_test.png"),
        "confusion_base_test": str(FIG_DIR / "confusion_base_test.png"),
        "context_adapter_heatmap": str(FIG_DIR / "context_adapter_heatmap.png"),
        "context_explicit_heatmap": str(FIG_DIR / "context_explicit_heatmap.png"),
        "non_time_controls": str(FIG_DIR / "non_time_controls.png"),
    }
    plot_time_layers(layer_metrics, Path(figure_paths["time_layer_metrics"]))
    plot_adapter_losses(adapter_summary["selected_losses"], Path(figure_paths["adapter_losses"]))
    plot_confusion(prediction_summary["explicit"]["test"]["confusion_matrix"], Path(figure_paths["confusion_explicit_test"]), "Explicit hidden decision head: test")
    plot_confusion(prediction_summary["adapter"]["test"]["confusion_matrix"], Path(figure_paths["confusion_adapter_test"]), "Time Adapter decision head: test")
    plot_confusion(prediction_summary["base"]["test"]["confusion_matrix"], Path(figure_paths["confusion_base_test"]), "No-time base hidden decision head: test")
    plot_context_heatmap(adapter_grid, Path(figure_paths["context_adapter_heatmap"]), "Context x time: adapter P(SUPPORT)-P(WAIT)")
    plot_context_heatmap(explicit_grid, Path(figure_paths["context_explicit_heatmap"]), "Context x time: explicit P(SUPPORT)-P(WAIT)")
    plot_non_time(non_time, Path(figure_paths["non_time_controls"]))

    summary = {
        "model_id": MODEL_ID,
        "device": str(device),
        "torch_version": torch.__version__,
        "dataset_manifest": manifest,
        "num_layers": int(explicit_hidden["train"].shape[1]),
        "hidden_size": int(explicit_hidden["train"].shape[2]),
        "selected_layer": best_layer,
        "layer_selection_metrics": layer_metrics,
        "time_prediction": time_prediction,
        "adapter": {
            **adapter_summary,
            "evaluation": adapter_eval,
        },
        "decision_head": head_summary,
        "prediction_metrics": prediction_summary,
        "random_vector_baseline": random_summary,
        "non_time_controls": non_time,
        "context_x_time": {
            "adapter_grid": adapter_grid,
            "explicit_grid": explicit_grid,
            "adapter_metrics": prediction_summary["adapter"]["test"],
            "explicit_metrics": prediction_summary["explicit"]["test"],
            "base_metrics": prediction_summary["base"]["test"],
        },
        "failures": failures,
        "figures": figure_paths,
        "artifacts": {
            "hidden_cache": str(hidden_dir),
            "time_direction": str(OUT_DIR / "time_direction.npy"),
            "time_adapter": str(OUT_DIR / "time_adapter_expanded.pt"),
            "decision_head": str(OUT_DIR / "decision_head.pkl"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
