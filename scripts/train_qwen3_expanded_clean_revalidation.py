from __future__ import annotations

from pathlib import Path

import train_qwen3_expanded_adapter as expanded


ROOT = Path(__file__).resolve().parents[1]

expanded.DATA_DIR = ROOT / "data/qwen3_context_time_expanded_clean_revalidation_v1"
expanded.OUT_DIR = ROOT / "artifacts/qwen3_expanded_training_clean_revalidation_v1"
expanded.FIG_DIR = ROOT / "output/figures/qwen3_expanded_training_clean_revalidation_v1"


if __name__ == "__main__":
    expanded.main()
