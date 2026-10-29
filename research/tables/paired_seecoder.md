# Paired comparison — B vs A

**Images:** 1111 paired by filename  
**Protocol:** test split, `center` crop, 256 px  
**Baseline checkpoint:** epoch 57  
**Ablation checkpoint:** epoch 58  
**Delta definition:** `delta = B - A`

| Metric | A | B | Δ mean | Δ median | Δ std | 95% CI on Δ mean | better on |
|---|---:|---:|---:|---:|---:|:---:|---:|
| PSNR (↑) | 28.9369 | 28.9907 | +0.0537 | +0.0603 | 0.3457 | [+0.0334, +0.0741] | 675/1111 |
| SSIM (↑) | 0.8661 | 0.8675 | +0.0015 | +0.0008 | 0.0097 | [+0.0009, +0.0020] | 703/1111 |
| LPIPS (↓) | 0.1356 | 0.1343 | -0.0013 | -0.0011 | 0.0090 | [-0.0018, -0.0007] | 671/1111 |

## How to read this

* The 95% CI is over TEST IMAGES for one fixed pair of checkpoints. It is NOT a training-run variance estimate.
* Each arm is a SINGLE training run. Run-to-run (seed) variance is unmeasured, so a significant per-image difference may still be seed noise rather than an effect of the module.
* Validation and test are the same split in this repository, so both checkpoints were selected on the data they are scored on.

`↑` higher is better, `↓` lower is better. A negative ΔLPIPS is an improvement.
