# Data is not included

This public repository contains no dataset rows, generated audio, or human recordings.

The retained builders create research data from explicit templates, deterministic label rules, and optional Qwen3-TTS speech synthesis. Workflows that generated semantic training examples with a text model are intentionally outside the public scope.

Expected generated directories include:

- `qwen3_context_time_phase1_3000/`
- `qwen3_context_time_expanded/`
- `omni_sequential_time_adapter/`
- `omni3b_sequential_v2/`
- `omni3b_delayed_backchannel_v2_qwen3tts_clean_0p6b/`

Start with the deterministic text dataset:

```powershell
python scripts/build_qwen3_phase1_dataset.py
```

Before distributing any regenerated data, record the builder commit, random seed, base-model revision, TTS voice/settings, split construction, and applicable licenses or consent. Generated voice audio should not be assumed redistributable merely because the generation script is public.
