# Paired comparison — F vs A

**Images:** 1111 paired by filename  
**Protocol:** test split, `center` crop, 256 px  
**Baseline checkpoint:** epoch 57  
**Ablation checkpoint:** epoch 56  
**Delta definition:** `delta = F - A`

| Metric | A | F | Δ mean | Δ median | Δ std | 95% CI on Δ mean | better on |
|---|---:|---:|---:|---:|---:|:---:|---:|
| PSNR (↑) | 28.9369 | 28.9361 | -0.0008 | -0.0039 | 0.3696 | [-0.0225, +0.0210] | 550/1111 |
| SSIM (↑) | 0.8661 | 0.8667 | +0.0006 | +0.0001 | 0.0101 | [+0.0000, +0.0012] | 578/1111 |
| LPIPS (↓) | 0.1356 | 0.1367 | +0.0011 | +0.0003 | 0.0085 | [+0.0006, +0.0016] | 528/1111 |

## How to read this

* The 95% CI is over TEST IMAGES for one fixed pair of checkpoints. It is NOT a training-run variance estimate.
* Each arm is a SINGLE training run. Run-to-run (seed) variance is unmeasured, so a significant per-image difference may still be seed noise rather than an effect of the module.
* Validation and test are the same split in this repository, so both checkpoints were selected on the data they are scored on.

`↑` higher is better, `↓` lower is better. A negative ΔLPIPS is an improvement.
