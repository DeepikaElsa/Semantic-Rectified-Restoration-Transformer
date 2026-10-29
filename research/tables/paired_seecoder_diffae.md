# Paired comparison — C vs A

**Images:** 1111 paired by filename  
**Protocol:** test split, `center` crop, 256 px  
**Baseline checkpoint:** epoch 57  
**Ablation checkpoint:** epoch 59  
**Delta definition:** `delta = C - A`

| Metric | A | C | Δ mean | Δ median | Δ std | 95% CI on Δ mean | better on |
|---|---:|---:|---:|---:|---:|:---:|---:|
| PSNR (↑) | 28.9369 | 28.9359 | -0.0010 | +0.0098 | 0.3194 | [-0.0198, +0.0178] | 574/1111 |
| SSIM (↑) | 0.8661 | 0.8668 | +0.0007 | +0.0002 | 0.0085 | [+0.0002, +0.0012] | 595/1111 |
| LPIPS (↓) | 0.1356 | 0.1358 | +0.0002 | -0.0001 | 0.0075 | [-0.0002, +0.0007] | 562/1111 |

## How to read this

* The 95% CI is over TEST IMAGES for one fixed pair of checkpoints. It is NOT a training-run variance estimate.
* Each arm is a SINGLE training run. Run-to-run (seed) variance is unmeasured, so a significant per-image difference may still be seed noise rather than an effect of the module.
* Validation and test are the same split in this repository, so both checkpoints were selected on the data they are scored on.

`↑` higher is better, `↓` lower is better. A negative ΔLPIPS is an improvement.
