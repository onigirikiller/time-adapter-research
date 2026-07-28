# Publication scope and provenance policy

This repository was assembled as a clean public-source snapshot rather than by publishing the original working tree or its Git history.

## Included

- core experiment, training, ablation, and local runtime code;
- deterministic template and label-rule builders;
- optional TTS synthesis code whose generated audio is not redistributed;
- aggregate research measurements and explicit limitations;
- no credentials or workstation-specific absolute paths.

## Excluded

- all dataset rows, generated speech, real recordings, and conversation logs;
- all LoRA, adapter, head, optimizer, cache, and base-model weights;
- workflows that generate semantic training examples with a text model, plus the dataset-dependent model and benchmark families built from them;
- internal report-production scripts and one-off delivery bundles;
- original Git metadata and commits, so removed material is not recoverable from public history;
- local toolchains, virtual environments, caches, and machine-transfer notes.

## Licensing boundary

The MIT license covers only the original source code and documentation in this repository. It does not relicense Qwen, Hugging Face components, TTS voices, generated audio, base-model weights, or any other third-party dependency.

In particular, `Qwen/Qwen2.5-Omni-3B` is governed by the Qwen Research License and is limited by its upstream terms to non-commercial research or evaluation unless a separate commercial license is obtained. No copy of that model or a derived checkpoint is included here. The other named model checkpoints currently referenced by the scripts use Apache-2.0 or MIT terms as recorded in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Anyone reproducing an experiment is responsible for reviewing the current upstream terms and the provenance of newly generated artifacts. Redistributing an environment, executable bundle, model, adapter, or generated voice requires a new artifact-level license review; this source-only audit does not cover such a future bundle.

## Result claims

Results are retained only as aggregate research measurements. They should be read with the dataset and hardware caveats in the README and should not be treated as independent benchmarks, safety claims, or production guarantees.
