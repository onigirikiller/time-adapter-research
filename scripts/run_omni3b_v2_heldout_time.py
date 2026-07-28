from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/omni3b_sequential_v2"
HIDDEN_DIR = OUT_DIR / "hidden_cache"
SUMMARY_PATH = OUT_DIR / "summary.json"
OUT_PATH = OUT_DIR / "heldout_time_summary.json"
HELDOUT_SECONDS = {0.75, 3.0}
STAGE = "extra"


def import_exp():
    path = ROOT / "scripts/run_omni3b_v2_experiment.py"
    spec = importlib.util.spec_from_file_location("omni3b_v2_exp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["omni3b_v2_exp"] = module
    spec.loader.exec_module(module)
    return module


exp = import_exp()


def sec(row):
    return round(float(row["silence_seconds"]), 2)


def subset_rows(rows, mask):
    return [row for row, keep in zip(rows, mask) if keep]


def subset_arr(arr, mask):
    return arr[np.asarray(mask, dtype=bool)]


def evaluate(rows, features, head, mean, std):
    pred, probs = exp.predict(head, mean, std, features)
    out = exp.metric(rows, pred, probs)
    out["sequence"] = exp.sequence_metrics(rows, pred)
    out["profile_accuracy"] = exp.profile_accuracy(rows, pred)
    out["per_second_accuracy"] = {}
    for t in sorted({sec(row) for row in rows}):
        m = np.array([sec(row) == t for row in rows], dtype=bool)
        mt = exp.metric(subset_rows(rows, m), pred[m], probs[m])
        out["per_second_accuracy"][str(t)] = {"accuracy": mt["accuracy"], "macro_f1": mt["macro_f1"], "n": int(m.sum())}
    return out


def main():
    splits, _ = exp.load_data()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    layer = int(summary["stage_results"][STAGE]["selected_layer"])
    stage_indices = {split: exp.stage_indices(splits[split], STAGE) for split in ["train", "validation", "test"]}
    rows = {split: [splits[split][int(i)] for i in stage_indices[split]] for split in stage_indices}
    hidden = {
        mode: {
            split: np.load(HIDDEN_DIR / f"{mode}_{split}.npy")[stage_indices[split], layer, :]
            for split in ["train", "validation", "test"]
        }
        for mode in ["no_time", "explicit"]
    }
    delta = {split: hidden["explicit"][split] - hidden["no_time"][split] for split in rows}
    known_mask = {split: np.array([sec(row) not in HELDOUT_SECONDS for row in rows[split]], dtype=bool) for split in rows}
    held_mask = {split: np.array([sec(row) in HELDOUT_SECONDS for row in rows[split]], dtype=bool) for split in rows}

    train_rows = subset_rows(rows["train"], known_mask["train"])
    val_rows = subset_rows(rows["validation"], known_mask["validation"])
    test_held_rows = subset_rows(rows["test"], held_mask["test"])
    test_known_rows = subset_rows(rows["test"], known_mask["test"])
    test_all_rows = rows["test"]

    adapter, adapter_summary = exp.v1.train_adapter(
        "heldout_time_multi_feature",
        exp.feature_matrix(train_rows, "multi"),
        subset_arr(delta["train"], known_mask["train"]),
        exp.feature_matrix(val_rows, "multi"),
        subset_arr(delta["validation"], known_mask["validation"]),
        exp.feature_matrix(test_held_rows, "multi"),
        subset_arr(delta["test"], held_mask["test"]),
        epochs=260,
    )

    adapter_delta = {
        split: exp.adapter_predict(adapter, exp.feature_matrix(rows[split], "multi"))
        for split in rows
    }
    oracle_features = {split: exp.decision_features(hidden["no_time"][split], delta[split]) for split in rows}
    adapter_features = {split: exp.decision_features(hidden["no_time"][split], adapter_delta[split]) for split in rows}

    train_x = np.vstack([
        subset_arr(adapter_features["train"], known_mask["train"]),
        subset_arr(oracle_features["train"], known_mask["train"]),
    ])
    train_y = np.concatenate([
        exp.y_labels(train_rows),
        exp.y_labels(train_rows),
    ])
    eval_features = {
        "correct_time_adapter": {
            "train": subset_arr(adapter_features["train"], known_mask["train"]),
            "validation": subset_arr(adapter_features["validation"], known_mask["validation"]),
            "test": subset_arr(adapter_features["test"], held_mask["test"]),
        },
        "oracle_explicit_delta": {
            "train": subset_arr(oracle_features["train"], known_mask["train"]),
            "validation": subset_arr(oracle_features["validation"], known_mask["validation"]),
            "test": subset_arr(oracle_features["test"], held_mask["test"]),
        },
    }
    eval_rows = {"train": train_rows, "validation": val_rows, "test": test_held_rows}
    head, mean, std, head_summary = exp.train_head(
        "heldout_time_adapter_head",
        train_x,
        train_y,
        eval_features,
        eval_rows,
        epochs=170,
    )

    no_head, no_mean, no_std, no_head_summary = exp.train_head(
        "heldout_no_time_context_head",
        subset_arr(hidden["no_time"]["train"], known_mask["train"]),
        exp.y_labels(train_rows),
        {"no_time_hidden": {
            "train": subset_arr(hidden["no_time"]["train"], known_mask["train"]),
            "validation": subset_arr(hidden["no_time"]["validation"], known_mask["validation"]),
            "test": subset_arr(hidden["no_time"]["test"], held_mask["test"]),
        }},
        eval_rows,
        epochs=170,
        lr=6e-4,
    )

    rng = np.random.default_rng(exp.SEED + 707)
    shuffled = adapter_delta["test"][rng.permutation(len(adapter_delta["test"]))]
    zero = np.zeros_like(adapter_delta["test"])
    cond_features = {
        "correct_time_adapter": adapter_features["test"],
        "oracle_explicit_delta": oracle_features["test"],
        "shuffled_time_adapter": exp.decision_features(hidden["no_time"]["test"], shuffled),
        "zero_vector": exp.decision_features(hidden["no_time"]["test"], zero),
    }

    def eval_subset(mask, row_subset):
        out = {}
        for name, feats in cond_features.items():
            out[name] = evaluate(row_subset, subset_arr(feats, mask), head, mean, std)
        out["no_time_hidden"] = evaluate(row_subset, subset_arr(hidden["no_time"]["test"], mask), no_head, no_mean, no_std)
        return out

    report = {
        "stage": STAGE,
        "heldout_seconds": sorted(HELDOUT_SECONDS),
        "selected_layer": layer,
        "counts": {
            "train_known": len(train_rows),
            "validation_known": len(val_rows),
            "test_known": len(test_known_rows),
            "test_heldout": len(test_held_rows),
            "test_all": len(test_all_rows),
        },
        "adapter": adapter_summary,
        "adapter_head": head_summary,
        "no_time_head": no_head_summary,
        "metrics": {
            "test_heldout_seconds": eval_subset(held_mask["test"], test_held_rows),
            "test_known_seconds": eval_subset(known_mask["test"], test_known_rows),
            "test_all_seconds": eval_subset(np.ones(len(test_all_rows), dtype=bool), test_all_rows),
        },
    }
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
