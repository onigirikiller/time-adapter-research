from __future__ import annotations

import csv
import json
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = ROOT / "artifacts/omni3b_generation_hook_v3"
OUT_DIR = ROOT / "artifacts/omni3b_diagnostics_v4"
PLOT_DIR = OUT_DIR / "plots"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def metric(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "backchannel_f1": float(f[1]),
        "wait_recall": float(r[0]),
        "support_recall": float(r[2]),
        "pred_counts": dict(Counter(LABELS[int(i)] for i in pred)),
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in range(3)},
    }


def build_rows(run_name, condition="correct_time_adapter"):
    all_rows = read_csv(V3_ROOT / run_name / "per_condition_results.csv")
    by_id = defaultdict(dict)
    for row in all_rows:
        by_id[row["id"]][row["condition"]] = row
    rows = []
    for row_id, m in by_id.items():
        if condition not in m or "no_time" not in m:
            continue
        r = m[condition]
        n = m["no_time"]
        item = {
            "id": row_id,
            "context_id": r["context_id"],
            "gold": r["gold_label"],
        }
        for prefix, src in [("c", r), ("n", n)]:
            for label in LABELS:
                low = label.lower()
                for name in ["avg_logprob", "logprob", "first_token_logprob", "prob", "raw_prob"]:
                    item[f"{prefix}_{low}_{name}"] = float(src[f"{low}_{name}"])
        for label in LABELS:
            low = label.lower()
            item[f"d_{low}_avg_logprob"] = item[f"c_{low}_avg_logprob"] - item[f"n_{low}_avg_logprob"]
            item[f"d_{low}_logprob"] = item[f"c_{low}_logprob"] - item[f"n_{low}_logprob"]
            item[f"d_{low}_first_token_logprob"] = item[f"c_{low}_first_token_logprob"] - item[f"n_{low}_first_token_logprob"]
        rows.append(item)
    return rows


FEATURE_SETS = {
    "correct_avg_logprobs": [f"c_{l.lower()}_avg_logprob" for l in LABELS],
    "correct_avg_plus_prior_delta": [f"c_{l.lower()}_avg_logprob" for l in LABELS] + [f"d_{l.lower()}_avg_logprob" for l in LABELS],
    "prior_delta_only": [f"d_{l.lower()}_avg_logprob" for l in LABELS],
    "all_lm_scores": (
        [f"c_{l.lower()}_{name}" for l in LABELS for name in ["avg_logprob", "logprob", "first_token_logprob", "prob", "raw_prob"]]
        + [f"n_{l.lower()}_{name}" for l in LABELS for name in ["avg_logprob", "logprob", "first_token_logprob", "prob", "raw_prob"]]
        + [f"d_{l.lower()}_{name}" for l in LABELS for name in ["avg_logprob", "logprob", "first_token_logprob"]]
    ),
}


def context_folds(rows, k=5):
    contexts = sorted({r["context_id"] for r in rows})
    folds = {c: i % k for i, c in enumerate(contexts)}
    return [folds[r["context_id"]] for r in rows]


def eval_run(run_name):
    rows = build_rows(run_name)
    y = np.array([LABEL_TO_ID[r["gold"]] for r in rows], dtype=np.int64)
    folds = np.array(context_folds(rows), dtype=np.int64)
    out = {}
    for feat_name, cols in FEATURE_SETS.items():
        preds = np.zeros_like(y)
        fold_metrics = []
        for fold in sorted(set(folds)):
            train = folds != fold
            test = folds == fold
            x_train = np.array([[r[c] for c in cols] for r in np.array(rows, dtype=object)[train]], dtype=np.float64)
            x_test = np.array([[r[c] for c in cols] for r in np.array(rows, dtype=object)[test]], dtype=np.float64)
            clf = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
            ])
            clf.fit(x_train, y[train])
            pred = clf.predict(x_test)
            preds[test] = pred
            fold_metrics.append(metric(y[test], pred))
        out[feat_name] = metric(y, preds)
        out[feat_name]["folds"] = fold_metrics
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": "Context-level 5-fold calibration heads trained only on LM label score features. This tests whether LM scores contain recoverable decision information.",
        "runs": {
            "full_l3_a4_all": eval_run("full_l3_a4_all"),
            "full_l8_a8_all": eval_run("full_l8_a8_all"),
        },
    }
    write_json(OUT_DIR / "lm_score_calibrator_cv.json", summary)

    for run_name, run in summary["runs"].items():
        names = list(run)
        vals = [run[n]["macro_f1"] for n in names]
        bvals = [run[n]["backchannel_f1"] for n in names]
        x = np.arange(len(names))
        plt.figure(figsize=(9, 4.5))
        plt.bar(x - 0.18, vals, width=0.36, label="macro F1")
        plt.bar(x + 0.18, bvals, width=0.36, label="BACKCHANNEL F1")
        plt.xticks(x, names, rotation=25, ha="right")
        plt.ylabel("F1")
        plt.title(f"LM score calibrator CV: {run_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"lm_score_calibrator_{run_name}.png", dpi=180)
        plt.close()
    print(OUT_DIR / "lm_score_calibrator_cv.json")


if __name__ == "__main__":
    main()
