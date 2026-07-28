from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/omni3b_sequential_v2"
V2_SUMMARY = ROOT / "artifacts/omni3b_sequential_v2/summary.json"
V3_ROOT = ROOT / "artifacts/omni3b_generation_hook_v3"
OUT_DIR = ROOT / "artifacts/omni3b_diagnostics_v4"
PLOT_DIR = OUT_DIR / "plots"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_from_labels(rows, pred_labels, prob_cols=None):
    y = np.array([LABEL_TO_ID[r["gold_label"] if "gold_label" in r else r["label"]] for r in rows], dtype=np.int64)
    p = np.array([LABEL_TO_ID[x] for x in pred_labels], dtype=np.int64)
    pr, re, f, s = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    out = {
        "rows": int(len(rows)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)),
        "backchannel_f1": float(f[LABEL_TO_ID["BACKCHANNEL"]]),
        "wait_recall": float(re[LABEL_TO_ID["WAIT"]]),
        "support_recall": float(re[LABEL_TO_ID["SUPPORT"]]),
        "per_class": {
            LABELS[i]: {"precision": float(pr[i]), "recall": float(re[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in range(3)
        },
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
        "pred_counts": dict(Counter(pred_labels)),
    }
    if prob_cols:
        probs = np.array([[float(r[c]) for c in prob_cols] for r in rows], dtype=np.float64)
        onehot = np.eye(3)[y]
        out["brier_score"] = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    return out


def sequence_metrics(rows, pred_labels):
    label_order = {"WAIT": 0, "BACKCHANNEL": 1, "SUPPORT": 2}
    by_context = defaultdict(list)
    for row, pred in zip(rows, pred_labels):
        by_context[row["context_id"]].append((row, pred))
    total = correct = exact = regressions = 0
    premature = delayed = 0
    for _, items in by_context.items():
        items = sorted(items, key=lambda x: float(x[0]["silence_seconds"]))
        gold = [r["gold_label"] if "gold_label" in r else r["label"] for r, _ in items]
        pred = [p for _, p in items]
        exact += int(gold == pred)
        for i, (g, p) in enumerate(zip(gold, pred)):
            total += 1
            correct += int(g == p)
            if label_order[p] > label_order[g]:
                premature += 1
            if label_order[p] < label_order[g]:
                delayed += 1
            if i and label_order[pred[i]] < label_order[pred[i - 1]]:
                regressions += 1
    return {
        "step_accuracy": float(correct / max(total, 1)),
        "exact_sequence_accuracy": float(exact / max(len(by_context), 1)),
        "premature_escalation_rate": float(premature / max(total, 1)),
        "delayed_support_rate": float(delayed / max(total, 1)),
        "regression_rate": float(regressions / max(total - len(by_context), 1)),
    }


def add_seq(metric, rows, pred):
    metric["sequence"] = sequence_metrics(rows, pred)
    return metric


def score_rule_predictions(rows, rule):
    if rule == "avg_logprob":
        cols = ["wait_avg_logprob", "backchannel_avg_logprob", "support_avg_logprob"]
    elif rule == "sum_logprob":
        cols = ["wait_logprob", "backchannel_logprob", "support_logprob"]
    elif rule == "first_token":
        cols = ["wait_first_token_logprob", "backchannel_first_token_logprob", "support_first_token_logprob"]
    else:
        raise ValueError(rule)
    pred = []
    for r in rows:
        vals = [float(r[c]) for c in cols]
        pred.append(LABELS[int(np.argmax(vals))])
    return pred


def prior_corrected_predictions(rows, no_time_by_id, rule):
    suffix = {
        "avg_logprob": "_avg_logprob",
        "sum_logprob": "_logprob",
        "first_token": "_first_token_logprob",
    }[rule]
    cols = [f"{label.lower()}{suffix}" for label in LABELS]
    pred = []
    margins = []
    for r in rows:
        base = no_time_by_id[r["id"]]
        vals = np.array([float(r[c]) - float(base[c]) for c in cols], dtype=np.float64)
        pred.append(LABELS[int(np.argmax(vals))])
        sorted_vals = np.sort(vals)
        margins.append(float(sorted_vals[-1] - sorted_vals[-2]))
    return pred, margins


def soft_distribution(rows_all):
    by = defaultdict(Counter)
    for r in rows_all:
        by[(r["profile"], float(r["silence_seconds"]))][r["label"]] += 1
    dist = {}
    for key, cnt in by.items():
        total = sum(cnt.values())
        dist[f"{key[0]}|{key[1]}"] = {label: cnt[label] / total for label in LABELS}
    return dist


def soft_eval(rows, probs, train_dist):
    kls = []
    briers = []
    top2 = []
    acceptable = []
    for r, p in zip(rows, probs):
        key = f"{r['profile']}|{float(r['silence_seconds'])}"
        q = np.array([train_dist.get(key, {}).get(label, 0.0) for label in LABELS], dtype=np.float64)
        if q.sum() <= 0:
            q = np.eye(3)[LABEL_TO_ID[r["gold_label"]]]
        q = q / q.sum()
        p = np.clip(np.array(p, dtype=np.float64), 1e-9, 1.0)
        p = p / p.sum()
        kls.append(float(np.sum(q * (np.log(q + 1e-9) - np.log(p)))))
        briers.append(float(np.sum((p - q) ** 2)))
        pred_order = np.argsort(p)[::-1]
        allowed = {i for i, v in enumerate(q) if v >= 0.20}
        if not allowed:
            allowed = {int(np.argmax(q))}
        top2.append(int(any(i in allowed for i in pred_order[:2])))
        acceptable.append(int(pred_order[0] in allowed))
    return {
        "kl_to_profile_time_distribution": float(np.mean(kls)),
        "soft_brier_score": float(np.mean(briers)),
        "top2_acceptable_accuracy": float(np.mean(top2)),
        "top1_acceptable_accuracy": float(np.mean(acceptable)),
    }


def ece(rows, probs):
    y = np.array([LABEL_TO_ID[r["gold_label"]] for r in rows], dtype=np.int64)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    bins = np.linspace(0, 1, 11)
    val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if not np.any(mask):
            continue
        val += float(np.mean(mask) * abs(np.mean(pred[mask] == y[mask]) - np.mean(conf[mask])))
    return val


def summarize_generation_run(run_name):
    run_dir = V3_ROOT / run_name
    rows = read_csv(run_dir / "per_condition_results.csv")
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    no_time_by_id = {r["id"]: r for r in by_cond["no_time"]}
    out = {"run_name": run_name, "conditions": {}}
    for cond, items in sorted(by_cond.items()):
        cond_out = {}
        for rule in ["avg_logprob", "sum_logprob", "first_token"]:
            pred = score_rule_predictions(items, rule)
            cond_out[rule] = add_seq(metric_from_labels(items, pred), items, pred)
        if cond != "no_time":
            for rule in ["avg_logprob", "sum_logprob", "first_token"]:
                pred, margins = prior_corrected_predictions(items, no_time_by_id, rule)
                m = add_seq(metric_from_labels(items, pred), items, pred)
                m["mean_prior_delta_margin"] = float(np.mean(margins))
                cond_out[f"prior_delta_{rule}"] = m
        pred_proxy = [r["proxy_label"] for r in items]
        cond_out["proxy_decision_head"] = add_seq(metric_from_labels(items, pred_proxy), items, pred_proxy)
        cond_out["lm_saved_generated_label"] = add_seq(metric_from_labels(items, [r["generated_label"] for r in items]), items, [r["generated_label"] for r in items])
        probs = np.array([[float(r[f"{label.lower()}_prob"]) for label in LABELS] for r in items], dtype=np.float64)
        cond_out["ece"] = ece(items, probs)
        out["conditions"][cond] = cond_out
    out["no_time_prior"] = {
        "mean_probabilities": {
            label: float(np.mean([float(r[f"{label.lower()}_prob"]) for r in by_cond["no_time"]]))
            for label in LABELS
        },
        "pred_counts": dict(Counter(r["generated_label"] for r in by_cond["no_time"])),
    }
    return out


def summarize_v2_scales():
    s = read_json(V2_SUMMARY)
    out = {}
    for stage, res in s["stage_results"].items():
        stage_out = {
            "selected_layer": res["selected_layer"],
            "counts": res.get("counts"),
            "conditions": {},
            "adapter": res.get("adapter", {}),
            "overfitting": res.get("overfitting", {}),
        }
        for cond, metric_blob in res["metrics"].items():
            test = metric_blob["test"]
            stage_out["conditions"][cond] = {
                "accuracy": test["accuracy"],
                "macro_f1": test["macro_f1"],
                "backchannel_f1": test["per_class"]["BACKCHANNEL"]["f1"],
                "wait_recall": test["per_class"]["WAIT"]["recall"],
                "support_recall": test["per_class"]["SUPPORT"]["recall"],
                "pred_counts": test["pred_counts"],
            }
        out[stage] = stage_out
    return out


def dataset_distribution():
    splits = {split: read_jsonl(DATA_DIR / f"{split}.jsonl") for split in ["train", "validation", "test"]}
    out = {}
    for split, rows in splits.items():
        lang = Counter("ja" if any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in r["fragment"]) else "en" for r in rows)
        out[split] = {
            "rows": len(rows),
            "contexts": len({r["context_id"] for r in rows}),
            "label_counts": dict(Counter(r["label"] for r in rows)),
            "profile_counts": dict(Counter(r["profile"] for r in rows)),
            "language_heuristic_counts": dict(lang),
            "time_counts": {str(k): v for k, v in sorted(Counter(float(r["silence_seconds"]) for r in rows).items())},
        }
    return out


def label_profile_time_soft_eval(run_name="full_l3_a4_all", condition="correct_time_adapter"):
    rows = [r for r in read_csv(V3_ROOT / run_name / "per_condition_results.csv") if r["condition"] == condition]
    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    dist = soft_distribution(train_rows)
    probs = [[float(r[f"{label.lower()}_prob"]) for label in LABELS] for r in rows]
    return soft_eval(rows, probs, dist)


def plots(summary):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    v2 = summary["v2_scale_effect"]
    stages = ["small", "medium", "large", "extra"]
    plt.figure(figsize=(9, 4.5))
    for cond in ["no_time_hidden", "correct_time_adapter", "zero_vector", "random_norm_matched", "non_time_numeric"]:
        plt.plot(stages, [v2[s]["conditions"][cond]["macro_f1"] for s in stages], marker="o", label=cond)
    plt.ylabel("Test macro F1")
    plt.title("Proxy decision head: dataset scale effect")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "v2_proxy_dataset_scale_macro_f1.png", dpi=180)
    plt.close()

    l3 = summary["generation_runs"]["full_l3_a4_all"]["conditions"]["correct_time_adapter"]
    names = ["avg_logprob", "sum_logprob", "first_token", "prior_delta_avg_logprob", "proxy_decision_head"]
    vals = [l3[n]["backchannel_f1"] for n in names]
    plt.figure(figsize=(9, 4.5))
    plt.bar(names, vals)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("BACKCHANNEL F1")
    plt.title("BACKCHANNEL bottleneck: LM scoring rules vs decision head")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "backchannel_f1_scoring_rules.png", dpi=180)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": "Offline diagnostics from existing Omni3B v2/v3 artifacts.",
        "generation_runs": {
            "full_l3_a4_all": summarize_generation_run("full_l3_a4_all"),
            "full_l8_a8_all": summarize_generation_run("full_l8_a8_all"),
        },
        "v2_scale_effect": summarize_v2_scales(),
        "dataset_distribution": dataset_distribution(),
        "soft_label_eval_main_correct": label_profile_time_soft_eval(),
        "proxy_hidden_position": {
            "source_script": "scripts/run_omni3b_v2_experiment.py",
            "forward_type": "teacher-forcing model.thinker forward",
            "output_hidden_states": True,
            "hidden_state_index": "outputs.hidden_states[1:]",
            "layer_semantics": "post-layer hidden state for each Thinker text layer",
            "token_position": "final token, h[0, -1, :]",
            "generate": False,
            "use_cache": "not explicitly disabled in v2 extraction",
        },
        "generation_hook_position": {
            "source_script": "scripts/run_omni3b_generation_hook_v3.py",
            "module": "model.thinker.model.layers[layer]",
            "hook_type": "forward_hook after layer output",
            "positions_tested_in_previous_v3": ["last_token", "all_tokens"],
            "main_full_run": {"layer": 3, "alpha": 4.0, "position": "all_tokens"},
            "strong_full_run": {"layer": 8, "alpha": 8.0, "position": "all_tokens"},
            "generate": False,
            "scoring": "teacher-forcing next-token logprob for label strings",
        },
    }
    plots(summary)
    summary["plots"] = {p.stem: str(p) for p in PLOT_DIR.glob("*.png")}
    write_json(OUT_DIR / "offline_summary.json", summary)
    print(OUT_DIR / "offline_summary.json")


if __name__ == "__main__":
    main()
