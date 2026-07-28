from __future__ import annotations

import sys
from pathlib import Path

import train_qwen3_phase1 as phase1


ROOT = Path(__file__).resolve().parents[1]

phase1.DATA_DIR = ROOT / "data/qwen3_context_time_phase1_3000_clean_revalidation_v1"
phase1.OUT_DIR = ROOT / "artifacts/qwen3_phase1_3000_clean_revalidation_v1"
phase1.FIG_DIR = ROOT / "output/figures/qwen3_phase1_3000_clean_revalidation_v1"


if __name__ == "__main__":
    phase1.main()
