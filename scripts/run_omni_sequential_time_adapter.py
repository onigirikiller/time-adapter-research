from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from qwen_omni_utils import process_mm_info
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("OMNI_DATA_DIR", str(ROOT / "data/omni_sequential_time_adapter")))
OUT_DIR = Path(os.environ.get("OMNI_OUT_DIR", str(ROOT / "artifacts/omni_sequential_time_adapter")))
FIG_DIR = Path(os.environ.get("OMNI_FIG_DIR", str(ROOT / "output/figures/omni_sequential_time_adapter")))
CACHE_DIR = ROOT / ".cache/huggingface"
MODEL_ID = os.environ.get("OMNI_MODEL_ID", "Qwen/Qwen2.5-Omni-3B")
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset():
    splits = {split: read_jsonl(DATA_DIR / f"{split}.jsonl") for split in ["train", "validation", "test"]}
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    return splits, manifest


def build_conversation(row: dict, mode: str, numeric_note: str | None = None):
    audio_path = str(Path(row["audio_path"]).resolve())
    base_instruction = (
        "Choose exactly one listener timing label: WAIT, BACKCHANNEL, or SUPPORT. "
        "WAIT means keep listening. BACKCHANNEL means give a short acknowledgement. "
        "SUPPORT means actively respond or help. Output only the label."
    )
    if mode == "audio_only":
        user_text = "The audio contains a partial user utterance and possible following silence. " + base_instruction
    elif mode == "explicit":
        f = row["features"]
        user_text = (
            f"ASR fragment: \"{row['fragment']}\"\n"
            "External timing features:\n"
            f"silence_elapsed={f['silence_elapsed']} seconds\n"
            f"delta_t={f['delta_t']} seconds\n"
            f"utterance_elapsed={f['utterance_elapsed']} seconds\n"
            f"is_user_speaking={f['is_user_speaking']}\n"
            f"asr_changed={f['asr_changed']}\n"
            + base_instruction
        )
    else:
        extra = f"\nUnrelated numeric note: {numeric_note}" if numeric_note else ""
        user_text = f"ASR fragment: \"{row['fragment']}\"{extra}\nNo timing values are provided.\n{base_instruction}"
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
        },
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}, {"type": "text", "text": user_text}]},
    ]


def move_inputs(inputs, device, dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
        else:
            moved[key] = value
    return moved


def load_model():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE_DIR),
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    return processor, model, model.device, dtype


def extract_hidden(processor, model, device, dtype, rows: list[dict], mode: str, out_path: Path | None, numeric_note: str | None = None):
    if out_path is not None and out_path.exists():
        return np.load(out_path)
    all_rows = []
    with torch.inference_mode():
        for row in rows:
            conv = build_conversation(row, mode, numeric_note=numeric_note)
            text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = move_inputs(inputs, device, dtype)
            outputs = model.thinker(**inputs, output_hidden_states=True, use_audio_in_video=False)
            layers = [h[0, -1, :].detach().float().cpu().numpy().astype(np.float32) for h in outputs.hidden_states[1:]]
            all_rows.append(np.stack(layers, axis=0))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    hidden = np.stack(all_rows, axis=0)
    if out_path is not None:
        np.save(out_path, hidden)
    return hidden


def y_seconds(rows):
    return np.array([row["silence_seconds"] for row in rows], dtype=np.float32)


def y_labels(rows):
    return np.array([LABEL_TO_ID[row["label"]] for row in rows], dtype=np.int64)


def feature_matrix(rows, kind: str):
    vals = []
    for row in rows:
        f = row["features"]
        if kind == "scalar":
            vals.append([np.log1p(f["silence_elapsed"])])
        else:
            vals.append(
                [
                    np.log1p(f["silence_elapsed"]),
                    f["delta_t"],
                    np.log1p(f["utterance_elapsed"]),
                    1.0 if f["is_user_speaking"] else 0.0,
                    1.0 if f["asr_changed"] else 0.0,
                ]
            )
    return np.array(vals, dtype=np.float32)


def select_layer(explicit_hidden, splits):
    metrics = []
    for layer in range(explicit_hidden["train"].shape[1]):
        reg = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        reg.fit(explicit_hidden["train"][:, layer, :], np.log1p(y_seconds(splits["train"])))
        pred = reg.predict(explicit_hidden["validation"][:, layer, :])
        true = np.log1p(y_seconds(splits["validation"]))
        corr = float(np.corrcoef(true, pred)[0, 1])
        metrics.append({"layer": layer, "validation_r2": float(r2_score(true, pred)), "validation_corr": corr})
    best = max(metrics, key=lambda row: row["validation_r2"])
    return int(best["layer"]), metrics


def time_prediction_metrics(explicit_features, splits, layer):
    out = {}
    train_x = explicit_features["train"][:, layer, :]
    train_y = np.log1p(y_seconds(splits["train"]))
    for split in ["train", "validation", "test"]:
        reg = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        reg.fit(train_x, train_y)
        true = np.log1p(y_seconds(splits[split]))
        pred = reg.predict(explicit_features[split][:, layer, :])
        out[split] = {"r2_log_seconds": float(r2_score(true, pred)), "corr_log_seconds": float(np.corrcoef(true, pred)[0, 1])}
    return out


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, hidden_size))

    def forward(self, x):
        return self.net(x)


def adapter_metrics(pred, true):
    mse = float(np.mean((pred - true) ** 2))
    zero = float(np.mean(true**2))
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    cos = np.divide(np.sum(pred * true, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
    return {"mse": mse, "zero_baseline_mse": zero, "mse_ratio_vs_zero": float(mse / max(zero, 1e-12)), "mean_cosine": float(np.mean(cos)), "median_cosine": float(np.median(cos))}


def train_adapter(name, x_train, y_train, x_val, y_val, x_test, y_test, epochs=260):
    set_seed(SEED + x_train.shape[1])
    adapter = FeatureAdapter(x_train.shape[1], y_train.shape[1])
    opt = torch.optim.AdamW(adapter.parameters(), lr=3e-3, weight_decay=1e-4)
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    xv = torch.tensor(x_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)
    xte = torch.tensor(x_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.float32)
    losses = []
    best_state, best_val = None, math.inf
    for epoch in range(epochs):
        adapter.train()
        opt.zero_grad()
        loss = torch.mean((adapter(xt) - yt) ** 2)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            adapter.eval()
            with torch.no_grad():
                val_loss = torch.mean((adapter(xv) - yv) ** 2)
                test_loss = torch.mean((adapter(xte) - yte) ** 2)
            row = {"epoch": epoch, "train_mse": float(loss.detach()), "validation_mse": float(val_loss.detach()), "test_mse": float(test_loss.detach())}
            losses.append(row)
            if row["validation_mse"] < best_val:
                best_val = row["validation_mse"]
                best_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    adapter.load_state_dict(best_state)
    adapter.eval()
    with torch.no_grad():
        val_pred = adapter(xv).detach().cpu().numpy()
        test_pred = adapter(xte).detach().cpu().numpy()
    return adapter, {"name": name, "losses": losses, "validation": adapter_metrics(val_pred, y_val), "test": adapter_metrics(test_pred, y_test)}


def adapter_predict(adapter, x):
    adapter.eval()
    with torch.no_grad():
        return adapter(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy().astype(np.float32)


class MLPHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, len(LABELS)))

    def forward(self, x):
        return self.net(x)


def make_decision_features(context, delta):
    return np.concatenate([context, delta], axis=1).astype(np.float32)


def fit_standardizer(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def metric(y_true, y_pred, probs=None):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
        "labels": LABELS,
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in range(len(LABELS))},
        "pred_counts": dict(Counter(LABELS[int(i)] for i in y_pred)),
    }
    if probs is not None:
        out["mean_probabilities"] = {LABELS[i]: float(np.mean(probs[:, i])) for i in range(len(LABELS))}
    return out


def train_head(features, splits, epochs=120):
    modes = ["explicit", "adapter_scalar", "adapter_multi"]
    x_train = np.vstack([features[m]["train"] for m in modes])
    y_train = np.concatenate([y_labels(splits["train"]) for _ in modes])
    mean, std = fit_standardizer(x_train)
    x_train = apply_standardizer(x_train, mean, std)
    model = MLPHead(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    curves = []
    best_state, best_score, best_epoch = None, -1.0, 0
    rng = np.random.default_rng(SEED)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x_tensor))
        for start in range(0, len(order), 128):
            idx = order[start : start + 128]
            logits = model(x_tensor[idx])
            loss = nn.functional.cross_entropy(logits, y_tensor[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 2 == 0 or epoch == epochs - 1:
            row = {"epoch": epoch}
            for mode in ["audio_only", "no_time", "explicit", "adapter_scalar", "adapter_multi"]:
                pred, probs = predict_head(model, mean, std, features[mode]["validation"])
                m = metric(y_labels(splits["validation"]), pred, probs)
                row[f"validation_{mode}_accuracy"] = m["accuracy"]
                row[f"validation_{mode}_macro_f1"] = m["macro_f1"]
                pred_t, probs_t = predict_head(model, mean, std, features[mode]["test"])
                mt = metric(y_labels(splits["test"]), pred_t, probs_t)
                row[f"test_{mode}_accuracy"] = mt["accuracy"]
                row[f"test_{mode}_macro_f1"] = mt["macro_f1"]
            score = row["validation_adapter_multi_macro_f1"]
            curves.append(row)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, mean, std, {"curves": curves, "selected_epoch": best_epoch, "selection_score": best_score}


def predict_head(model, mean, std, x):
    model.eval()
    xs = torch.tensor(apply_standardizer(x, mean, std), dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(xs), dim=-1).cpu().numpy()
    return np.argmax(probs, axis=1), probs


def evaluate_modes(model, mean, std, features, splits):
    out = {}
    for mode in ["audio_only", "no_time", "explicit", "adapter_scalar", "adapter_multi"]:
        out[mode] = {}
        for split in ["train", "validation", "test"]:
            pred, probs = predict_head(model, mean, std, features[mode][split])
            out[mode][split] = metric(y_labels(splits[split]), pred, probs)
    pred_exp, _ = predict_head(model, mean, std, features["explicit"]["test"])
    pred_ad, _ = predict_head(model, mean, std, features["adapter_multi"]["test"])
    out["explicit_adapter_multi_test_agreement"] = float(np.mean(pred_exp == pred_ad))
    return out


def random_baseline(model, mean, std, context_test, adapter_delta_test, rows):
    rng = np.random.default_rng(SEED)
    norms = np.linalg.norm(adapter_delta_test, axis=1).reshape(-1, 1)
    vals = []
    for trial in range(5):
        direction = rng.normal(size=context_test.shape[1]).astype(np.float32)
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        feats = make_decision_features(context_test, direction.reshape(1, -1) * norms)
        pred, probs = predict_head(model, mean, std, feats)
        vals.append({"trial": trial, "metrics": metric(y_labels(rows), pred, probs)})
    return {"trials": vals, "mean_accuracy": float(np.mean([v["metrics"]["accuracy"] for v in vals])), "mean_macro_f1": float(np.mean([v["metrics"]["macro_f1"] for v in vals]))}


def context_grid(rows, pred, probs):
    grid = []
    for row, pr, p in zip(rows, pred, probs):
        grid.append(
            {
                "context_id": row["context_id"],
                "profile": row["profile"],
                "fragment": row["fragment"],
                "seconds": row["silence_seconds"],
                "gold": row["label"],
                "prediction": LABELS[int(pr)],
                "support_minus_wait": float(p[LABEL_TO_ID["SUPPORT"]] - p[LABEL_TO_ID["WAIT"]]),
                "backchannel_minus_wait": float(p[LABEL_TO_ID["BACKCHANNEL"]] - p[LABEL_TO_ID["WAIT"]]),
                "probabilities": {LABELS[i]: float(p[i]) for i in range(len(LABELS))},
            }
        )
    return grid


def sequential_transitions(rows, grid):
    by_context = defaultdict(list)
    for item in grid:
        by_context[item["context_id"]].append(item)
    out = []
    for context_id, items in by_context.items():
        items = sorted(items, key=lambda x: x["seconds"])
        out.append(
            {
                "context_id": context_id,
                "profile": items[0]["profile"],
                "fragment": items[0]["fragment"],
                "gold_sequence": [x["gold"] for x in items],
                "prediction_sequence": [x["prediction"] for x in items],
                "seconds": [x["seconds"] for x in items],
            }
        )
    return out


def failure_examples(rows, pred, probs, limit=20):
    failures = []
    for row, pr, p in zip(rows, pred, probs):
        if LABELS[int(pr)] != row["label"]:
            failures.append(
                {
                    "context_id": row["context_id"],
                    "profile": row["profile"],
                    "seconds": row["silence_seconds"],
                    "fragment": row["fragment"],
                    "gold": row["label"],
                    "prediction": LABELS[int(pr)],
                    "confidence": float(np.max(p)),
                    "probabilities": {LABELS[i]: float(p[i]) for i in range(len(LABELS))},
                }
            )
    failures.sort(key=lambda x: x["confidence"], reverse=True)
    return failures[:limit]


def plot_curves(curves, path):
    plt.figure(figsize=(8, 4.6))
    epochs = [r["epoch"] for r in curves]
    for mode in ["audio_only", "no_time", "explicit", "adapter_scalar", "adapter_multi"]:
        plt.plot(epochs, [r[f"validation_{mode}_macro_f1"] for r in curves], label=f"{mode} val")
        plt.plot(epochs, [r[f"test_{mode}_macro_f1"] for r in curves], linestyle="--", label=f"{mode} test")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Omni sequential decision head curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_confusion(cm, path, title):
    arr = np.array(cm)
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


def plot_heatmap(grid, key, path, title):
    profiles = sorted({g["profile"] for g in grid})
    seconds = sorted({g["seconds"] for g in grid})
    data = np.full((len(profiles), len(seconds)), np.nan)
    for i, profile in enumerate(profiles):
        for j, sec in enumerate(seconds):
            vals = [g[key] for g in grid if g["profile"] == profile and g["seconds"] == sec]
            if vals:
                data[i, j] = float(np.mean(vals))
    plt.figure(figsize=(8.4, 4.8))
    im = plt.imshow(data, aspect="auto", cmap="PiYG", vmin=-1, vmax=1)
    plt.colorbar(im, label=key)
    plt.xticks(range(len(seconds)), [str(s) for s in seconds])
    plt.yticks(range(len(profiles)), profiles)
    plt.xlabel("silence_elapsed")
    plt.title(title)
    for i in range(len(profiles)):
        for j in range(len(seconds)):
            if not np.isnan(data[i, j]):
                plt.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_transitions(transitions, path):
    label_y = {label: i for i, label in enumerate(LABELS)}
    plt.figure(figsize=(9, 5))
    for tr in transitions:
        y = [label_y[x] for x in tr["prediction_sequence"]]
        plt.plot(tr["seconds"], y, marker="o", alpha=0.7, label=tr["profile"])
    plt.yticks(range(len(LABELS)), LABELS)
    plt.xlabel("silence_elapsed")
    plt.ylabel("Predicted label")
    plt.title("Sequential label transitions on held-out contexts")
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-epochs", type=int, default=260)
    parser.add_argument("--head-epochs", type=int, default=120)
    args = parser.parse_args()
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hidden_dir = OUT_DIR / "hidden_cache"
    hidden_dir.mkdir(exist_ok=True)

    splits, manifest = load_dataset()
    processor, model, device, dtype = load_model()
    print(f"Loaded {MODEL_ID} on {device}", flush=True)

    hidden = {mode: {} for mode in ["audio_only", "no_time", "explicit"]}
    for mode in hidden:
        for split, rows in splits.items():
            hidden[mode][split] = extract_hidden(processor, model, device, dtype, rows, mode, hidden_dir / f"{mode}_{split}.npy")
            print(f"Hidden {mode} {split}: {hidden[mode][split].shape}", flush=True)

    layer, layer_metrics = select_layer(hidden["explicit"], splits)
    time_prediction = time_prediction_metrics(hidden["explicit"], splits, layer)
    print(f"Selected layer {layer}", flush=True)

    no_time = {split: hidden["no_time"][split][:, layer, :] for split in splits}
    audio_only = {split: hidden["audio_only"][split][:, layer, :] for split in splits}
    explicit = {split: hidden["explicit"][split][:, layer, :] for split in splits}
    deltas = {split: explicit[split] - no_time[split] for split in splits}

    scalar_adapter, scalar_summary = train_adapter(
        "scalar_silence",
        feature_matrix(splits["train"], "scalar"),
        deltas["train"],
        feature_matrix(splits["validation"], "scalar"),
        deltas["validation"],
        feature_matrix(splits["test"], "scalar"),
        deltas["test"],
        args.adapter_epochs,
    )
    multi_adapter, multi_summary = train_adapter(
        "multi_feature",
        feature_matrix(splits["train"], "multi"),
        deltas["train"],
        feature_matrix(splits["validation"], "multi"),
        deltas["validation"],
        feature_matrix(splits["test"], "multi"),
        deltas["test"],
        args.adapter_epochs,
    )
    torch.save({"scalar": scalar_adapter.state_dict(), "multi": multi_adapter.state_dict(), "layer": layer, "model_id": MODEL_ID}, OUT_DIR / "time_adapters_omni_sequential.pt")

    scalar_delta = {split: adapter_predict(scalar_adapter, feature_matrix(splits[split], "scalar")) for split in splits}
    multi_delta = {split: adapter_predict(multi_adapter, feature_matrix(splits[split], "multi")) for split in splits}
    zero_delta = {split: np.zeros_like(deltas[split]) for split in splits}
    features = {
        "audio_only": {split: make_decision_features(audio_only[split], zero_delta[split]) for split in splits},
        "no_time": {split: make_decision_features(no_time[split], zero_delta[split]) for split in splits},
        "explicit": {split: make_decision_features(no_time[split], deltas[split]) for split in splits},
        "adapter_scalar": {split: make_decision_features(no_time[split], scalar_delta[split]) for split in splits},
        "adapter_multi": {split: make_decision_features(no_time[split], multi_delta[split]) for split in splits},
    }
    head, mean, std, head_summary = train_head(features, splits, args.head_epochs)
    with (OUT_DIR / "decision_head_omni_sequential.pkl").open("wb") as f:
        pickle.dump({"state_dict": head.state_dict(), "mean": mean, "std": std, "selected_epoch": head_summary["selected_epoch"]}, f)

    metrics = evaluate_modes(head, mean, std, features, splits)
    adapter_pred, adapter_probs = predict_head(head, mean, std, features["adapter_multi"]["test"])
    explicit_pred, explicit_probs = predict_head(head, mean, std, features["explicit"]["test"])
    no_time_pred, no_time_probs = predict_head(head, mean, std, features["no_time"]["test"])
    audio_pred, audio_probs = predict_head(head, mean, std, features["audio_only"]["test"])

    random_summary = random_baseline(head, mean, std, no_time["test"], multi_delta["test"], splits["test"])

    # A lightweight non-time numeric baseline: use text numeric-note hidden direction at the selected layer.
    non_time = {}
    for name, note in {"kg": "[5kg]", "m": "[5m]", "score": "[5 points]", "yen": "[5 yen]"}.items():
        probe_rows = splits["train"][:32]
        probe_hidden = extract_hidden(processor, model, device, dtype, probe_rows, "no_time", out_path=None, numeric_note=note)
        direction = np.mean(probe_hidden[:, layer, :] - no_time["train"][: len(probe_rows)], axis=0)
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        norms = np.linalg.norm(multi_delta["test"], axis=1).reshape(-1, 1)
        feats = make_decision_features(no_time["test"], direction.reshape(1, -1) * norms)
        pred, probs = predict_head(head, mean, std, feats)
        non_time[name] = {"cosine_to_multi_adapter_mean": float(np.dot(direction, np.mean(multi_delta["test"], axis=0)) / max(np.linalg.norm(np.mean(multi_delta["test"], axis=0)), 1e-12)), "metrics": metric(y_labels(splits["test"]), pred, probs)}

    grid_adapter = context_grid(splits["test"], adapter_pred, adapter_probs)
    grid_explicit = context_grid(splits["test"], explicit_pred, explicit_probs)
    transitions = sequential_transitions(splits["test"], grid_adapter)
    failures = {
        "adapter_multi": failure_examples(splits["test"], adapter_pred, adapter_probs),
        "explicit": failure_examples(splits["test"], explicit_pred, explicit_probs),
        "no_time": failure_examples(splits["test"], no_time_pred, no_time_probs),
        "audio_only": failure_examples(splits["test"], audio_pred, audio_probs),
    }

    figures = {
        "head_curves": str(FIG_DIR / "head_curves.png"),
        "confusion_adapter_multi": str(FIG_DIR / "confusion_adapter_multi.png"),
        "confusion_audio_only": str(FIG_DIR / "confusion_audio_only.png"),
        "support_wait_heatmap": str(FIG_DIR / "support_wait_heatmap.png"),
        "backchannel_wait_heatmap": str(FIG_DIR / "backchannel_wait_heatmap.png"),
        "explicit_support_wait_heatmap": str(FIG_DIR / "explicit_support_wait_heatmap.png"),
        "transitions": str(FIG_DIR / "sequential_transitions.png"),
    }
    plot_curves(head_summary["curves"], Path(figures["head_curves"]))
    plot_confusion(metrics["adapter_multi"]["test"]["confusion_matrix"], Path(figures["confusion_adapter_multi"]), "Omni adapter-multi test")
    plot_confusion(metrics["audio_only"]["test"]["confusion_matrix"], Path(figures["confusion_audio_only"]), "Omni audio-only test")
    plot_heatmap(grid_adapter, "support_minus_wait", Path(figures["support_wait_heatmap"]), "Adapter multi SUPPORT-WAIT")
    plot_heatmap(grid_adapter, "backchannel_minus_wait", Path(figures["backchannel_wait_heatmap"]), "Adapter multi BACKCHANNEL-WAIT")
    plot_heatmap(grid_explicit, "support_minus_wait", Path(figures["explicit_support_wait_heatmap"]), "Explicit SUPPORT-WAIT")
    plot_transitions(transitions, Path(figures["transitions"]))

    summary = {
        "model_id": MODEL_ID,
        "dataset_manifest": manifest,
        "selected_layer": layer,
        "layer_selection_metrics": layer_metrics,
        "time_prediction": time_prediction,
        "adapters": {"scalar": scalar_summary, "multi": multi_summary},
        "head": head_summary,
        "metrics": metrics,
        "random_vector_baseline": random_summary,
        "non_time_numeric_baseline": non_time,
        "explicit_adapter_multi_test_agreement": metrics["explicit_adapter_multi_test_agreement"],
        "context_x_time": {"adapter_multi_grid": grid_adapter, "explicit_grid": grid_explicit},
        "sequential_transitions": transitions,
        "failures": failures,
        "figures": figures,
        "artifacts": {
            "dataset": str(DATA_DIR),
            "hidden_cache": str(hidden_dir),
            "time_adapters": str(OUT_DIR / "time_adapters_omni_sequential.pt"),
            "decision_head": str(OUT_DIR / "decision_head_omni_sequential.pkl"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
