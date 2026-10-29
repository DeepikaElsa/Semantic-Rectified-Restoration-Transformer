# Worked Numerical Examples

Small hand-checkable examples for the non-obvious formulas. Every number below
is verified by `tests/test_numerical_examples.py`.

---

## 1. Rectified Flow — interpolation and flow loss

**Given**

```
z0 = [1, 2]
z1 = [5, 6]
t  = 0.25
```

**Step 1 — interpolate.** `z_t = (1-t)·z0 + t·z1`

```
z_t = 0.75 · [1, 2] + 0.25 · [5, 6]
    = [0.75, 1.50] + [1.25, 1.50]
    = [2.00, 3.00]
```

**Step 2 — target velocity.** `v* = z1 - z0`

```
v* = [5, 6] - [1, 2] = [4, 4]
```

Note `v*` does not depend on `t` — the path is straight. That is exactly why
few Euler steps suffice.

**Step 3 — flow loss.** Suppose the network predicts `v_theta = [3.8, 4.2]`.

Squared L2 (sum convention):

```
L_flow = ||v_theta - v*||²
       = (3.8 - 4)² + (4.2 - 4)²
       = (-0.2)²    + (0.2)²
       = 0.04 + 0.04
       = 0.08
```

Mean convention (what `F.mse_loss` returns, and what the code uses):

```
L_flow = 0.08 / 2 = 0.04
```

**Step 4 — sanity check on the endpoints.**

```
t = 0:  z_t = 1·[1,2] + 0·[5,6] = [1,2] = z0   ✓
t = 1:  z_t = 0·[1,2] + 1·[5,6] = [5,6] = z1   ✓
```

**Step 5 — one Euler step.** With `N = 4` steps, `dt = 0.25`, starting at
`z = z0 = [1,2]`, and supposing `v_theta(z_0, 0) = [4, 4]` (the ideal value):

```
z_1 = z_0 + dt · v = [1,2] + 0.25·[4,4] = [2,3]
```

which equals `z_t` at `t = 0.25` from Step 1. With a perfectly learned constant
velocity, Euler integration is *exact* for rectified flow — 4 steps land
precisely on `z1`:

```
z_4 = [1,2] + 4 · 0.25 · [4,4] = [1,2] + [4,4] = [5,6] = z1   ✓
```

---

## 2. L1 restoration loss

**Given** a 2×2 single-channel prediction and target:

```
x_hat = [[0.9, 0.2],      x = [[1.0, 0.0],
         [0.4, 0.7]]           [0.5, 1.0]]
```

**Absolute differences**

```
|0.9-1.0| = 0.1     |0.2-0.0| = 0.2
|0.4-0.5| = 0.1     |0.7-1.0| = 0.3
```

**L1**

```
L_L1 = (0.1 + 0.2 + 0.1 + 0.3) / 4 = 0.7 / 4 = 0.175
```

---

## 3. PSNR

Using the same tensors:

```
MSE = (0.1² + 0.2² + 0.1² + 0.3²) / 4
    = (0.01 + 0.04 + 0.01 + 0.09) / 4
    = 0.15 / 4
    = 0.0375

PSNR = 10 · log10( 1² / 0.0375 )
     = 10 · log10( 26.6667 )
     = 10 · 1.42597
     = 14.2597 dB
```

A perfect prediction gives `MSE = 0` and `PSNR = ∞`; the code clamps `MSE` to
`1e-10` during training and returns `inf` in evaluation.

---

## 4. MDTA attention shape — why the matmul order matters

Take a deliberately tiny case: `B=1, C=4, H=W=2`, `h=2`, so `d = C/h = 2` and
`HW = 4`.

After reshaping, `Q, K, V` each have shape `(1, 2, 2, 4)` = `(B, h, d, HW)`.

**Correct (channel attention):**

```
Q · K^T :  (1,2,2,4) · (1,2,4,2)  ->  (1,2,2,2)     = (B, h, d, d)
```

The `HW=4` axis is contracted away. Matrix has `d² = 4` entries.

**Incorrect (what the old code did):**

```
Q^T · K :  (1,2,4,2) · (1,2,2,4)  ->  (1,2,4,4)     = (B, h, HW, HW)
```

The `d=2` axis is contracted; the matrix has `HW² = 16` entries.

Both look harmless at this size. Scale to `H=W=256`, `C=48`, `h=1`:

| | matrix shape | elements | fp32 bytes |
|---|---|---|---|
| correct | `48 × 48` | 2,304 | 9.2 KB |
| incorrect | `65536 × 65536` | 4,294,967,296 | **17.2 GB** |

Hence the historical `~34 GB` allocation failure (multiple intermediates plus
the softmax output). The ratio is `(HW/d)² = (65536/48)² ≈ 1.86 million`.

---

## 5. LCR consistency loss — collapse resistance

Let the projection head output (before normalization) be `u_pred` and `u_ref`.
After L2 normalization both lie on the unit sphere, so

```
L_LCR = ||P_pred - P_ref||²
      = ||P_pred||² + ||P_ref||² - 2·<P_pred, P_ref>
      = 1 + 1 - 2·cos(angle)
      = 2·(1 - cos(angle))
```

Consequences:

* identical latents → `cos = 1` → `L_LCR = 0`
* orthogonal → `cos = 0` → `L_LCR = 2`
* opposite → `cos = -1` → `L_LCR = 4` (maximum)

The loss depends **only on direction**, so the network cannot reduce it by
scaling latents toward zero — the standard collapse shortcut for unnormalized
consistency losses.
