# SRRT — IEEE paper source

`srrt_ieee.tex` — IEEE conference format (`IEEEtran`, `[conference]` option).
All six ablation arms are trained, evaluated and written up. Every number in
the paper is measured; nothing is estimated.

## Compiling

No LaTeX toolchain is installed in this repository's environment, so the file
has **not** been compiled here. It passes `scripts/check_paper.py` (balanced
environments and braces, all `\ref` resolve, `\cite`/`\bibitem` consistent, all
figure files present) — which is not the same as a successful build.

**Overleaf:** create a project from the *IEEE Conference* template, upload
`srrt_ieee.tex` and the whole `figures/` directory, compile. `IEEEtran.cls` is
already there.

**Locally**, with TeX Live or MiKTeX:

```bash
pdflatex srrt_ieee
pdflatex srrt_ieee      # second pass resolves \ref and citation numbers
```

The bibliography is a `thebibliography` block, so no BibTeX pass is needed.
TikZ is required for Fig. 1 (it ships with both distributions).

## Figures

Regenerate everything with:

```bash
python scripts/make_paper_figures.py              # all four data figures
python scripts/make_paper_figures.py --skip-qual  # plots only, no GPU
```

| Figure | File | Source |
|---|---|---|
| 1 Architecture | inline TikZ | hand-drawn in the `.tex` |
| 2 Qualitative | `fig_qualitative.pdf` | checkpoints A and E, 3 test images |
| 3 Per-module effect | `fig_ablation.pdf` | all six `results/*/summary.json` |
| 4 Convergence | `fig_convergence.pdf` | all six `logs/*/train_log.csv` |
| 5 Flow divergence | `fig_divergence.pdf` | pre- and post-fix flow logs |

Each figure writes a `.json` sidecar recording exactly which artifacts it was
built from. `fig_qualitative.json` records that the magnified window is chosen
to maximise the baseline-minus-SRRT error — favourable by construction, and
disclosed as such in the caption.

## Source of every number

| Claim | Artifact |
|---|---|
| Six-arm PSNR/SSIM/LPIPS table | `results/*/summary.json` |
| Paired Δ, 95% CI, win/loss, Wilcoxon p | `research/tables/paired_*.json` |
| Zero-gate equivalence, all exactly 0.0 | `results/ablation_seecoder/zero_gate.json` |
| SeeCoder disconnection, 0/44 gradients | `research_notes/VALIDATION_STATUS.md` §4 |
| Flow divergence: 24.098 → 14.527 → 6.606 | `logs/seecoder_diffae_lcr_flow_200EP_ACCIDENTAL/train_log.csv` (first run block) |
| Pre-fix E and F evaluations | `results/*_PREFIX_BROKEN/summary.json` |
| Partial SeeCoder run, 28.496 at 40/60 | `logs/seecoder_41EP_INCOMPLETE/train_log.csv` |
| 200-epoch E reaching 29.65 at epoch 88 | `logs/seecoder_diffae_lcr_flow_200EP_ACCIDENTAL/train_log.csv` |
| Inference timing, eval VRAM | `results/*/summary.json` |

GFLOPs are reported as unmeasured because neither `thop` nor `fvcore` was
installed.

## Before submitting

1. **Fill in the author block** — still placeholders.
2. **Compile and check the page count.** `scripts/check_paper.py` estimates
   ~5 pages against the 6-page limit, but only a real build settles it.
3. Consider the open items in Section VI: single seed per arm is the binding
   limitation on every claim in the paper.

## History

`srrt_ieee_PRE_RESULTS.tex.bak` is the 2026-09-06 draft, written before the
corrected runs, when E and F were still diverged and marked inadmissible. Kept
because its framing of the failure analysis is more detailed than the space in
the current draft allows.
