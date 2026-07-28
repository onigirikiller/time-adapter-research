from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import train_qwen3_phase1 as phase1
from analyze_qwen3_clean_revalidation import LABELS, RUNS, load_features, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/qwen3_clean_revalidation_analysis_v1/mlp_head_ablation.json"


def eval_modes(head, real_features, splits, device):
    return {
        mode: phase1.metric_at(head, real_features, splits, "test", mode, device)
        for mode in ["base", "explicit", "adapter"]
    }


def train_variant(name: str, real_features, train_source: str, splits, epochs: int, device):
    if train_source == "combined":
        train_features = real_features
    else:
        train_features = {
            "base": real_features[train_source],
            "explicit": real_features[train_source],
            "adapter": real_features[train_source],
        }
    head = phase1.train_head(name, "mlp", "hard", train_features, splits, epochs, device)
    return eval_modes(head, real_features, splits, device)


def main():
    phase1.set_seed(20260630)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for run_key in ["v5_dirty", "v5_clean"]:
        data, _summary, _base_h, _deltas, _adapter_v, features = load_features(run_key)
        results[run_key] = {}
        for train_source in ["base", "explicit", "adapter", "combined"]:
            results[run_key][f"train_{train_source}"] = train_variant(
                f"{run_key}_{train_source}",
                features,
                train_source,
                data,
                epochs=80,
                device=device,
            )
    write_json(OUT, {"device": str(device), "results": results})
    print(OUT)


if __name__ == "__main__":
    main()
