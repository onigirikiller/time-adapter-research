# Experiment map

Run scripts from the repository root. They are research snapshots rather than a stable package, and they write generated data or results below ignored `data/`, `artifacts/`, and `output/` directories.

## 1. Hidden time direction

Start here. `run_time_direction_experiment.py` measures whether explicit time can be linearly recovered from transformer hidden states and tests direction-vector and adapter interventions.

```powershell
python scripts/run_time_direction_experiment.py --models Qwen/Qwen2.5-0.5B-Instruct
```

## 2. Context × time text experiments

| Purpose | Script |
|---|---|
| Build deterministic train/validation/test rows | `build_qwen3_phase1_dataset.py` |
| Train and evaluate the first text Time Adapter | `train_qwen3_phase1.py` |
| Build the expanded dataset | `build_qwen3_expanded_dataset.py` |
| Train the expanded adapter | `train_qwen3_expanded_adapter.py` |
| Build leakage-resistant revalidation splits | `build_qwen3_clean_revalidation_dataset.py` |
| Analyze clean revalidation results | `analyze_qwen3_clean_revalidation.py` |

## 3. Audio prefix and sequential timing

| Purpose | Script |
|---|---|
| Build templated/TTS sequential audio | `generate_omni_sequential_dataset.py` |
| First audio-prefix Time Adapter | `run_omni_sequential_time_adapter.py` |
| Larger held-out-context dataset | `generate_omni3b_v2_dataset.py` |
| Multi-stage proxy-head experiment | `run_omni3b_v2_experiment.py` |
| Held-out timing check | `run_omni3b_v2_heldout_time.py` |

## 4. Generation-path diagnostics

`run_omni3b_generation_hook_v3.py`, `run_omni3b_diagnostic_offline_v4.py`, `run_omni3b_diagnostic_model_v4.py`, and `run_omni3b_lm_score_calibrator_v4.py` isolate the gap between a strong proxy head and a weak base-LM action-token head.

## 5. Direct action-token training

`run_omni3b_single_token_lora_v1.py` trains short `/W`, `/B`, and `/S` action tokens. `run_omni3b_control_response_lora_v1.py` extends the target with short templated backchannel/support text. Delayed-backchannel dataset variants are built by the retained `train_omni3b_delayed_backchannel_v1.py` and `train_omni3b_delayed_backchannel_v2.py` snapshots.

## 6. Runtime experiments

`run_control_response_lora_pseudorealtime_v1.py` evaluates the label-first runtime. `run_control_response_lora_webui_v1.py` exposes the same idea through a local FastAPI/WebSocket microphone UI. The WebUI expects compatible artifacts under the paths declared at the top of the script.

The server has no authentication. Keep the default `127.0.0.1` binding unless you add an appropriate security boundary.
