# Paired comparison — D vs A

**Images:** 1111 paired by filename  
**Protocol:** test split, `center` crop, 256 px  
**Baseline checkpoint:** epoch 57  
**Ablation checkpoint:** epoch 59  
**Delta definition:** `delta = D - A`

| Metric | A | D | Δ mean | Δ median | Δ std | 95% CI on Δ mean | better on |
|---|---:|---:|---:|---:|---:|:---:|---:|
| PSNR (↑) | 28.9369 | 29.0138 | +0.0769 | +0.0631 | 0.3147 | [+0.0584, +0.0954] | 678/1111 |
| SSIM (↑) | 0.8661 | 0.8683 | +0.0022 | +0.0011 | 0.0092 | [+0.0017, +0.0028] | 722/1111 |
| LPIPS (↓) | 0.1356 | 0.1322 | -0.0033 | -0.0025 | 0.0079 | [-0.0038, -0.0028] | 807/1111 |

## How to read this

* The 95% CI is over TEST IMAGES for one fixed pair of checkpoints. It is NOT a training-run variance estimate.
* Each arm is a SINGLE training run. Run-to-run (seed) variance is unmeasured, so a significant per-image difference may still be seed noise rather than an effect of the module.
* Validation and test are the same split in this repository, so both checkpoints were selected on the data they are scored on.

`↑` higher is better, `↓` lower is better. A negative ΔLPIPS is an improvement.
