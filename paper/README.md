# Research preprint

**Timing Before Talking: Time Adapters for Low-Latency Turn-Taking in Spoken Language Models**  
Towa Yoshida, Rissho University  
ORCID: <https://orcid.org/0009-0002-1715-0219>

- Reviewed PDF: [`timing_before_talking.pdf`](timing_before_talking.pdf)
- LaTeX source: [`main.tex`](main.tex)
- Bibliography: [`references.bib`](references.bib)
- Reproducible figures: [`make_figures.py`](make_figures.py)
- Copy-ready arXiv metadata: [`ARXIV_SUBMISSION.md`](ARXIV_SUBMISSION.md)

## Build

From this directory:

```powershell
python make_figures.py  # requires ReportLab
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The paper is an experimental preprint, not a peer-reviewed publication. Its quantitative claims are limited to the generated/templated evaluation described in the paper and a small local pseudo-realtime check. The repository intentionally excludes model weights, generated datasets, recordings, and private evaluation material.

## Suggested arXiv metadata

- Primary category: `cs.CL`
- Cross-list candidates: `cs.HC`, `eess.AS`
- Comments: `Research preprint; source-only research release; 5 pages including references and appendix.`
- Code: <https://github.com/onigirikiller/time-adapter-research>

The prepared source archive in the GitHub Release contains only the files needed by arXiv. The author must complete arXiv's account, authorship, license, and endorsement steps personally.
