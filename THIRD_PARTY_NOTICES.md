# Third-party license inventory

Last reviewed: 2026-07-28

This repository is a source-only research release. It does not vendor Python packages, model weights, adapters, audio, FFmpeg, libsndfile, CUDA components, or other third-party binaries. Listing or importing a dependency does not place that dependency under this repository's MIT license.

This inventory records the upstream terms relevant to the pinned direct dependencies and model identifiers currently referenced by the source. Transitive dependencies and future upstream releases must be reviewed again when producing a redistributed environment or binary bundle.

## Material model restriction

`Qwen/Qwen2.5-Omni-3B` is licensed under the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/main/LICENSE). Its model materials are limited to non-commercial research or evaluation unless a separate commercial license is obtained from Alibaba Cloud. Redistribution of those materials or derivatives has additional license-copy, modification-notice, and attribution requirements. The upstream license also requires a prominent `Built with Qwen` or `Improved using Qwen` notice when a model created, trained, fine-tuned, or improved using the materials is distributed or made available.

This repository does not include Qwen2.5-Omni-3B, a derivative model, or any trained LoRA/adapter/checkpoint. The original source code remains MIT-licensed, but the MIT grant does not override the model restriction. Commercial use of the code together with Qwen2.5-Omni-3B requires a separate model license or a suitably licensed replacement model.

Research experiments described in this repository were built with Qwen.

## Referenced models

| Model identifier | Upstream license | Included here | Practical consequence |
|---|---|---:|---|
| [`Qwen/Qwen2.5-Omni-3B`](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | Qwen Research License | No | Non-commercial research/evaluation only without a separate commercial license; derivative redistribution has additional notice requirements. |
| [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache-2.0 | No | Downloaded separately; retain upstream license/NOTICE when redistributing the model. |
| [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Apache-2.0 | No | Downloaded separately; retain upstream license/NOTICE when redistributing the model. |
| [`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | Apache-2.0 | No | Downloaded separately; review generated-audio provenance and applicable voice/personality rights before redistribution. |
| [`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Apache-2.0 | No | Same boundary as the 0.6B checkpoint. |
| [`gpt2`](https://huggingface.co/openai-community/gpt2) | MIT | No | Downloaded separately. |

## Direct Python dependencies

The following is a registry/source review of the versions pinned in `requirements.txt`, plus PyTorch, which the installation instructions require separately.

| Package | Pinned version | Declared upstream license |
|---|---:|---|
| `accelerate` | 1.12.0 | Apache-2.0 |
| `fastapi` | 0.138.0 | MIT |
| `imageio` | 2.37.3 | BSD-2-Clause |
| `imageio-ffmpeg` | 0.6.0 | BSD-2-Clause for the Python wrapper; bundled FFmpeg has separate terms described below |
| `librosa` | 0.11.0 | ISC |
| `matplotlib` | 3.11.0 | Matplotlib/PSF-based license |
| `numpy` | 2.4.4 | BSD-3-Clause plus permissively licensed bundled components/notices |
| `peft` | 0.19.1 | Apache-2.0 |
| `pillow` | 12.2.0 | MIT-CMU |
| `pywin32` | 312 | PSF-2.0 |
| `qwen-omni-utils` | 0.0.9 | Apache-2.0 |
| `qwen-tts` | 0.1.1 | Apache-2.0 |
| `scikit-learn` | 1.9.0 | BSD-3-Clause |
| `scipy` | 1.18.0 | BSD-3-Clause plus bundled third-party notices |
| `soundfile` | 0.14.0 | BSD-3-Clause for the Python wrapper; platform wheels may bundle LGPL-2.1 `libsndfile` |
| `transformers` | commit `11ed2ff4` | Apache-2.0 |
| `triton-windows` | 3.7.1.post27 | MIT |
| `uvicorn` | 0.49.0 | BSD-3-Clause |
| `websockets` | 16.0 | BSD-3-Clause |
| `torch` | installed separately | BSD-3-Clause, with additional third-party notices in binary distributions |

These licenses do not prevent this repository's original source from being published under MIT because none of the packages is copied into this repository. Apache, BSD, ISC, PSF, and MIT-family dependencies remain under their own terms when installed.

## Binary redistribution cautions

- The [`imageio-ffmpeg`](https://github.com/imageio/imageio-ffmpeg) wheels include an FFmpeg executable. FFmpeg is primarily LGPL-2.1-or-later, but a particular build can become GPL-covered depending on enabled components. Do not copy that executable into a release or application bundle without recording the exact build configuration and satisfying its corresponding license/source obligations.
- The [`soundfile`](https://pypi.org/project/soundfile/) platform wheels can include `libsndfile`, which is LGPL-2.1. Installing the wheel for local use does not relicense this repository, but redistributing the wheel/library in a product requires preserving the applicable notices and LGPL compliance path.
- CUDA, NVIDIA drivers/toolkits, Microsoft runtimes, and any binary components pulled by PyTorch or Triton are not distributed by this repository. A future packaged application must audit the exact binaries it ships rather than relying on this source-only inventory.

## Scope and maintenance

This is an engineering license inventory, not legal advice. It establishes that no identified direct dependency prevents this source-only repository from using MIT for its original code. It does not establish ownership of code copied without attribution, patent clearance, trademark rights, dataset consent, generated-voice rights, or compliance of a future binary/model release.
