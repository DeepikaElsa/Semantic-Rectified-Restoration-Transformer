# Paired comparison — E vs A

**Images:** 1111 paired by filename  
**Protocol:** test split, `center` crop, 256 px  
**Baseline checkpoint:** epoch 57  
**Ablation checkpoint:** epoch 59  
**Delta definition:** `delta = E - A`

| Metric | A | E | Δ mean | Δ median | Δ std | 95% CI on Δ mean | better on |
|---|---:|---:|---:|---:|---:|:---:|---:|
| PSNR (↑) | 28.9369 | 29.0588 | +0.1219 | +0.1020 | 0.3451 | [+0.1016, +0.1422] | 756/1111 |
| SSIM (↑) | 0.8661 | 0.8694 | +0.0034 | +0.0015 | 0.0100 | [+0.0028, +0.0040] | 761/1111 |
| LPIPS (↓) | 0.1356 | 0.1325 | -0.0030 | -0.0019 | 0.0082 | [-0.0035, -0.0025] | 757/1111 |

## How to read this

* The 95% CI is over TEST IMAGES for one fixed pair of checkpoints. It is NOT a training-run variance estimate.
* Each arm is a SINGLE training run. Run-to-run (seed) variance is unmeasured, so a significant per-image difference may still be seed noise rather than an effect of the module.
* Validation and test are the same split in this repository, so both checkpoints were selected on the data they are scored on.

`↑` higher is better, `↓` lower is better. A negative ΔLPIPS is an improvement.
