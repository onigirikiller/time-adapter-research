from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_qwen3_phase1 as phase1
import train_qwen3_expanded_adapter as expanded


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/qwen3_clean_revalidation_analysis_v1"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
FW_L = "\uff08"
FW_R = "\uff09"
BETSU_MARKER = "\u5225\u306e\u5834\u9762\u3068\u3057\u3066"


RUNS = {
    "v4_dirty": {
        "data": ROOT / "data/qwen3_context_time_expanded",
        "artifact": ROOT / "artifacts/qwen3_expanded_training",
        "summary": ROOT / "artifacts/qwen3_expanded_training/summary.json",
        "trainer": "expanded",
    },
    "v4_clean": {
        "data": ROOT / "data/qwen3_context_time_expanded_clean_revalidation_v1",
        "artifact": ROOT / "artifacts/qwen3_expanded_training_clean_revalidation_v1",
        "summary": ROOT / "artifacts/qwen3_expanded_training_clean_revalidation_v1/summary.json",
        "trainer": "expanded",
    },
    "v5_dirty": {
        "data": ROOT / "data/qwen3_context_time_phase1_3000",
        "artifact": ROOT / "artifacts/qwen3_phase1_3000",
        "summary": ROOT / "artifacts/qwen3_phase1_3000/summary.json",
        "trainer": "phase1",
    },
    "v5_clean": {
        "data": ROOT / "data/qwen3_context_time_phase1_3000_clean_revalidation_v1",
        "artifact": ROOT / "artifacts/qwen3_phase1_3000_clean_revalidation_v1",
        "summary": ROOT / "artifacts/qwen3_phase1_3000_clean_revalidation_v1/summary.json",
        "trainer": "phase1",
    },
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def y(rows: list[dict]) -> np.ndarray:
    return np.array([LABEL_TO_ID[row["label"]] for row in rows], dtype=np.int64)


def seconds(rows: list[dict]) -> np.ndarray:
    return np.array([float(row["seconds"]) for row in rows], dtype=np.float32)


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "per_class": {
            label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }


def contamination_audit(data_dir: Path) -> dict:
    out = {}
    for split in ["train", "validation", "test"]:
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        frags = [row["fragment"] for row in rows]
        out[split] = {
            "rows": len(rows),
            "split_colon_markers": sum(any(x in f for x in ["train:", "validation:", "test:"]) for f in frags),
            "fullwidth_parentheses": sum(FW_L in f or FW_R in f for f in frags),
            "ascii_parentheses": sum("(" in f or ")" in f for f in frags),
            "duplicate_marker": sum(BETSU_MARKER in f for f in frags),
            "long_digit_run": sum(bool(re.search(r"\d{3,}", f)) for f in frags),
            "unique_fragments": len(set(frags)),
        }
    frag_sets = {split: {row["fragment"] for row in read_jsonl(data_dir / f"{split}.jsonl")} for split in ["train", "validation", "test"]}
    out["overlap"] = {
        "train_validation": len(frag_sets["train"] & frag_sets["validation"]),
        "train_test": len(frag_sets["train"] & frag_sets["test"]),
        "validation_test": len(frag_sets["validation"] & frag_sets["test"]),
    }
    return out


def load_adapter(run_key: str, summary: dict, artifact: Path):
    trainer = RUNS[run_key]["trainer"]
    if trainer == "phase1":
        ckpt = torch.load(
            artifact / "time_adapter_phase1.pt",
            map_location="cpu",
            weights_only=True,
        )
        adapter = phase1.TimeAdapter(summary["hidden_size"], ckpt["width"])
    else:
        ckpt = torch.load(
            artifact / "time_adapter_expanded.pt",
            map_location="cpu",
            weights_only=True,
        )
        adapter = expanded.TimeAdapter(summary["hidden_size"], ckpt["width"])
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    return adapter


def adapter_vectors(adapter, sec: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return adapter(torch.tensor(sec, dtype=torch.float32)).detach().cpu().numpy().astype(np.float32)


def load_features(run_key: str):
    cfg = RUNS[run_key]
    data = {split: read_jsonl(cfg["data"] / f"{split}.jsonl") for split in ["train", "validation", "test"]}
    summary = read_json(cfg["summary"])
    artifact = cfg["artifact"]
    layer = int(summary["selected_layer"])
    hidden_dir = artifact / "hidden_cache"
    explicit_h = {split: np.load(hidden_dir / f"explicit_{split}.npy")[:, layer, :] for split in data}
    base_h = {split: np.load(hidden_dir / f"base_{split}.npy")[:, layer, :] for split in data}
    deltas = {split: explicit_h[split] - base_h[split] for split in data}
    adapter = load_adapter(run_key, summary, artifact)
    adapter_v = {split: adapter_vectors(adapter, seconds(data[split])) for split in data}
    zeros = {split: np.zeros_like(adapter_v[split]) for split in data}
    features = {
        "base": {split: np.concatenate([base_h[split], zeros[split]], axis=1).astype(np.float32) for split in data},
        "explicit": {split: np.concatenate([base_h[split], deltas[split]], axis=1).astype(np.float32) for split in data},
        "adapter": {split: np.concatenate([base_h[split], adapter_v[split]], axis=1).astype(np.float32) for split in data},
    }
    return data, summary, base_h, deltas, adapter_v, features


def train_probe(x_train: np.ndarray, y_train: np.ndarray):
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced", solver="lsqr")
    pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
    pipe.fit(x_train, y_train)
    return pipe


def probe_ablation(run_key: str) -> dict:
    data, summary, _base_h, _deltas, _adapter_v, features = load_features(run_key)
    y_train = y(data["train"])
    y_test = y(data["test"])
    train_sets = {
        "train_base_only": features["base"]["train"],
        "train_explicit_only": features["explicit"]["train"],
        "train_adapter_only": features["adapter"]["train"],
        "train_explicit_plus_adapter": np.vstack([features["explicit"]["train"], features["adapter"]["train"]]),
    }
    y_train_sets = {
        "train_base_only": y_train,
        "train_explicit_only": y_train,
        "train_adapter_only": y_train,
        "train_explicit_plus_adapter": np.concatenate([y_train, y_train]),
    }
    out = {}
    for train_name, x_train in train_sets.items():
        probe = train_probe(x_train, y_train_sets[train_name])
        out[train_name] = {}
        for eval_mode in ["base", "explicit", "adapter"]:
            pred = probe.predict(features[eval_mode]["test"])
            out[train_name][f"eval_{eval_mode}"] = metric(y_test, pred)
    return out


def variance_stats(rows: list[dict], deltas: np.ndarray, adapter_v: np.ndarray) -> dict:
    sec = seconds(rows)
    out = {}
    for name, arr in [("explicit_delta", deltas), ("adapter_vector", adapter_v)]:
        by_sec = defaultdict(list)
        for i, s in enumerate(sec):
            by_sec[float(s)].append(i)
        distances = []
        variances = []
        for s, idxs in by_sec.items():
            x = arr[idxs]
            centroid = x.mean(axis=0, keepdims=True)
            d = np.linalg.norm(x - centroid, axis=1)
            distances.extend(d.tolist())
            variances.append(float(np.mean(np.sum((x - centroid) ** 2, axis=1))))
        norms = np.linalg.norm(arr, axis=1)
        out[name] = {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "mean_within_second_l2_to_centroid": float(np.mean(distances)),
            "median_within_second_l2_to_centroid": float(np.median(distances)),
            "mean_within_second_variance": float(np.mean(variances)),
        }
    denom = np.linalg.norm(deltas, axis=1) * np.linalg.norm(adapter_v, axis=1)
    cos = np.divide(np.sum(deltas * adapter_v, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
    out["explicit_adapter_alignment"] = {
        "mean_cosine": float(np.mean(cos)),
        "median_cosine": float(np.median(cos)),
        "mean_l2_distance": float(np.mean(np.linalg.norm(deltas - adapter_v, axis=1))),
    }
    return out


def type_metrics(rows: list[dict], summary: dict, mode: str) -> dict:
    grid = summary["context_x_time"][f"{mode}_grid"]
    pred_by_id = {row["id"]: row["prediction"] for row in grid}
    y_true = []
    y_pred = []
    by_type = defaultdict(lambda: [[], []])
    for row in rows:
        pred = pred_by_id.get(row["id"])
        if pred is None:
            continue
        yt = LABEL_TO_ID[row["label"]]
        yp = LABEL_TO_ID[pred]
        y_true.append(yt)
        y_pred.append(yp)
        by_type[row["time_expression_type"]][0].append(yt)
        by_type[row["time_expression_type"]][1].append(yp)
    return {k: metric(np.array(v[0]), np.array(v[1])) for k, v in by_type.items() if v[0]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"runs": {}, "probe_ablation": {}, "variance": {}, "time_expression_type_metrics": {}}
    for run_key, cfg in RUNS.items():
        summary = read_json(cfg["summary"])
        audit = contamination_audit(cfg["data"])
        result["runs"][run_key] = {
            "data_dir": str(cfg["data"]),
            "artifact_dir": str(cfg["artifact"]),
            "contamination_audit": audit,
            "selected_layer": summary.get("selected_layer"),
            "selected_head": summary.get("selected_head"),
        }
        if "selected_head_metrics" in summary:
            result["runs"][run_key]["selected_head_metrics"] = {
                mode: summary["selected_head_metrics"][mode]["test"] for mode in ["base", "explicit", "adapter"]
            }
        elif "prediction_metrics" in summary:
            result["runs"][run_key]["selected_head_metrics"] = {
                mode: summary["prediction_metrics"][mode]["test"] for mode in ["base", "explicit", "adapter"]
            }
        if "adapter" in summary:
            result["runs"][run_key]["adapter_test_reconstruction"] = summary["adapter"].get("evaluation", {}).get("test")

    for run_key in ["v5_dirty", "v5_clean"]:
        data, summary, _base_h, deltas, adapter_v, _features = load_features(run_key)
        result["variance"][run_key] = {
            split: variance_stats(data[split], deltas[split], adapter_v[split]) for split in ["train", "validation", "test"]
        }
        result["probe_ablation"][run_key] = probe_ablation(run_key)
        result["time_expression_type_metrics"][run_key] = {
            "explicit": type_metrics(data["test"], summary, "explicit"),
            "adapter": type_metrics(data["test"], summary, "adapter"),
        }

    write_json(OUT / "qwen3_clean_revalidation_analysis_summary.json", result)
    print(OUT / "qwen3_clean_revalidation_analysis_summary.json")


if __name__ == "__main__":
    main()
