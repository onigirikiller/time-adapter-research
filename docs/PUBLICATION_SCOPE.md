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

The MIT license covers only the original source code and documentation in this repository. It does not relicense Qwen, Hugging Face components, TTS voices, generated audio, base-model weights, or any other third-party dependency. Anyone reproducing an experiment is responsible for reviewing those upstream terms and the provenance of newly generated artifacts.

## Result claims

Results are retained only as aggregate research measurements. They should be read with the dataset and hardware caveats in the README and should not be treated as independent benchmarks, safety claims, or production guarantees.
