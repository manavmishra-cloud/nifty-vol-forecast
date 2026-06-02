# Paper — arXiv preprint draft

This directory contains the LaTeX source for the paper accompanying the
`nifty-vol-forecast` codebase.

**Working title:** *Sequence Models, Not Tabular ML, Beat HAR-RV: A Walk-Forward
Benchmark of Volatility Forecasts on NIFTY-50*

**Target venue:** arXiv `q-fin.ST` (Statistical Finance) — workshop paper
length (~8-10 pages).

**Status:** Draft v1 — methodology and results sections substantive, intro
and conclusion narrative complete, discussion still light.

## Files

```
paper/
├── README.md          # this file
├── main.tex           # paper source
└── references.bib     # bibliography
```

## How to build the PDF

### Option 1: Overleaf (easiest)

1. Create a free account at https://overleaf.com
2. Create a new project → Upload Project → upload `main.tex` and `references.bib`
3. Click "Recompile" — the PDF appears in the preview pane
4. Iterate on the source, recompile, download PDF when ready

### Option 2: Local LaTeX (Linux)

```bash
sudo apt install texlive-latex-recommended texlive-fonts-extra texlive-bibtex-extra biber
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The `pdflatex` → `bibtex` → `pdflatex` × 2 dance is needed for citations
to resolve properly.

### Option 3: Docker (no system install)

```bash
docker run --rm -v $PWD:/data -w /data texlive/texlive:latest \
  latexmk -pdf -interaction=nonstopmode main.tex
```

## Sections

| Section | Status | Word count (approx) |
|---|---|---|
| Abstract | ✓ Complete | 200 |
| 1. Introduction | ✓ Complete | 700 |
| 2. Related Work | ✓ Complete | 400 |
| 3. Data | ✓ Complete | 350 |
| 4. Methodology | ✓ Complete | 700 |
| 5. Results | ✓ Complete with real numbers | 500 |
|  5.1 Main comparison | ✓ | -- |
|  5.2 Crisis regimes | ✓ | -- |
|  5.3 Robustness (sensitivity) | ✓ added | 250 |
| 6. Discussion | ✓ Updated to reference robustness | 350 |
| 7. Conclusion | ✓ Complete | 300 |
| References | ✓ 23 entries (all verified) | -- |

Total: ~3,750 words, expected PDF length ~9-11 pages.

## Next iteration

Recommended additions before submission:

1. **Robustness checks** — sensitivity to seq_len, refit_freq, initial train size
2. **Multi-horizon results** — repeat the comparison at 5- and 22-day horizons
3. **Regime-conditional analysis** — split out crisis vs calm periods
4. **Confusion plot / forecast trace figures** in Section 5
5. **Compute / wall-clock comparison** — fairness check on cost
6. **Better discussion** — why sequence models specifically help

## arXiv submission checklist

When ready to upload:

- [ ] Title and abstract finalized
- [ ] Author affiliation (currently VIT only — add Numiv if approved)
- [ ] All claims supported by numbers in the results table
- [ ] All figures included and rendering at print resolution
- [ ] Bibliography sorted, all `\cite{...}` keys resolved
- [ ] Code repo link in the paper resolves
- [ ] License confirmed (MIT for code; arXiv preprint license for paper itself)
- [ ] Acknowledgments section if applicable
- [ ] Run a grammar/spell pass (Grammarly free tier, or `aspell -c main.tex`)
- [ ] arXiv account created and validated for q-fin submission
- [ ] An endorser from arXiv contacted (first-time submitters need one)
