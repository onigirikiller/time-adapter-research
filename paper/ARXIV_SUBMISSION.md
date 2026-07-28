# arXiv submission metadata

This file is a copy-ready submission sheet. The paper is prepared for arXiv, but the author must personally complete arXiv's account, authorship, license, and endorsement steps.

## Title

Timing Before Talking: Time Adapters for Low-Latency Turn-Taking in Spoken Language Models

## Authors

Towa Yoshida

## Affiliation and contact

- Rissho University, Tokyo, Japan
- onigirikiller@proton.me
- <https://orcid.org/0009-0002-1715-0219>

## Categories

- Primary: `cs.CL` (Computation and Language)
- Possible cross-list: `cs.HC` (Human-Computer Interaction)
- Possible cross-list: `eess.AS` (Audio and Speech Processing)

## Abstract

Spoken language models can produce fluent responses, but deciding *when* to speak remains a separate systems problem. Running full generation at every audio tick is expensive, while a model that ignores elapsed silence may interrupt, respond late, or miss opportunities for a short backchannel. We study a compact Time Adapter that maps explicit streaming-time features to a residual injected into an audio-language model hidden layer. A high-frequency decision path scores three single-token actions—wait (`/W`), backchannel (`/B`), and support/respond (`/S`)—and invokes text or speech generation only when needed. On a generated sequential benchmark of 600 utterances evaluated at ten silence durations, a proxy decision head reaches 0.989 macro F1. However, feeding the same residual to the frozen base language-model head reaches only 0.428 macro F1, revealing an objective-alignment failure rather than a lack of temporal information. Training the action-token logits directly with low-rank adaptation raises macro F1 to 0.998 in an audio-only setting; a control-token-plus-short-response objective also reaches 0.998. In a four-case local pseudo-realtime check on an RTX 4090, action scoring has 309.1 ms p95 latency and 99.3% of 137 ticks complete within 500 ms, excluding asynchronous response generation. These results are engineering evidence from synthetic and templated evaluation, not an independent natural-dialogue benchmark. We release source code, ablations, and this paper to make both the positive and negative findings reproducible.

## Comments

Research preprint; source-only research release; 5 pages including references and appendix. Code: <https://github.com/onigirikiller/time-adapter-research>

## License choice

For the author's stated fully open release goal, `CC BY 4.0` is the recommended arXiv submission choice. The author should read and select the license personally in the arXiv submission form. This does not change third-party model terms: Qwen2.5-Omni-3B remains subject to the Qwen Research License and is not redistributed.
