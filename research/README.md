# `research/` — publication figure and table pipeline

Everything here is produced from **measured experiment data**. No script in this
pipeline invents, interpolates, or estimates a number; unmeasured quantities are
rendered as `not run` / `--` and stay that way until an experiment produces them.

## Layout

```
research/
├── figures/          training curves, error maps, frequency, ablation, efficiency
│   ├── features/     channel-mean activations and per-channel maps
│   └── attention/    MDTA channel-attention stats, SRA spatial gate
├── tables/           CSV + LaTeX tables, profile.json
├── qualitative/      per-image restoration comparisons
│   └── failure_cases/
├── feature_maps/     reserved for exported raw feature tensors
├── attention_maps/   reserved for exported raw attention tensors
├── reports/          written analyses
└── FIGURE_PLAN.md    per-figure purpose, data needs, and availability
```

## Scripts

| Script | Produces | Needs |
|---|---|---|
| `plot_training_curves.py` | 6-panel training diagnostics | a `train_log.csv` |
| `generate_qualitative_grid.py` | restoration grids + zooms + failure cases | ≥1 checkpoint |
| `visualize_error_maps.py` | abs / squared / normalized error maps | ≥1 checkpoint |
| `visualize_frequency.py` | FFT spectra + radial profiles | ≥1 checkpoint |
| `visualize_features.py` | channel-mean and per-channel activations | 1 checkpoint |
| `visualize_attention.py` | MDTA channel attention, SRA gate | 1 checkpoint |
| `compare_ablation_results.py` | ablation bar charts | `results/<ablation>/summary.json` |
| `generate_paper_tables.py` | `.csv` + `.tex` tables | same |
| `profile_models.py` | params / FLOPs / latency / VRAM | nothing (architecture only) |

Shared behaviour lives in [`utils/research.py`](../utils/research.py): typography,
colour scheme, log reading, sample selection, error/FFT definitions, and provenance.

## Integrity guarantees built into the code

1. **Read-only logs.** `read_train_log` snapshots the CSV and drops a torn trailing
   line, so figures can be generated against a *running* experiment without
   touching it.
2. **Model-independent sample selection.** `select_image_ids(len, n, seed)` sees no
   model and no metric, and is *nested* in `n`, so the same seed yields the same
   (or a superset of the) images across every script and every model.
3. **Ground-truth-driven zoom windows.** The zoom region is the highest
   gradient-energy window of the **ground truth**, never of a prediction, so no
   model can select the region it happens to win on.
4. **Shared error colour scales.** Error maps across models use one `vmax`, so a
   darker map means less error. `normalized` mode is per-image and is labelled
   on the figure as not comparable across panels.
5. **Protocol guard.** `assert_comparable` refuses to build a chart or table that
   mixes evaluation protocols (e.g. 256px centre crops with full images) — that
   difference alone would look like a multi-dB improvement.
6. **Provenance sidecars.** Every figure gets a `<name>.json` recording experiment,
   checkpoint, epoch, dataset, split, image IDs, seed, resolution, crop mode,
   normalization, git hash and argv.

## Interpretation limits stated on the figures themselves

- Frequency recovery does **not** prove better restoration (hallucination and
  ringing also raise high-frequency energy; the magnitude spectrum drops phase).
- Feature maps are qualitative activation magnitudes, not evidence of semantics.
- MDTA attention is **channel** attention (d×d), never a spatial map — it is never
  reshaped or overlaid on an image.
- Attention visualizations do not establish causality; the ablation ladder does.

## Current status

The Restormer baseline is still training. **No SRRT ablation has been run**, so
every comparison figure is baseline-only and nothing here supports a claim that
SRRT improves on Restormer. See [`FIGURE_PLAN.md`](FIGURE_PLAN.md) for what is
blocked and on what.
