# SRRT — Mathematical Formulation

## 1. Notation

| Symbol | Meaning |
|---|---|
| `y` | degraded input image |
| `x` | ground-truth clean image |
| `x_hat` | predicted (restored) image |
| `X` | a feature tensor |
| `B, C, H, W` | batch, channels, height, width |
| `HW` | number of spatial positions, `H·W` |
| `Q, K, V` | query, key, value |
| `h` | number of attention heads |
| `d` | channels per head, `d = C/h` |
| `alpha` | learnable attention temperature |
| `theta` | learnable parameters |
| `z` | latent representation |
| `z0` | degraded / initial latent |
| `z1` | clean / target latent |
| `z_t` | interpolated latent at flow time `t` |
| `t` | flow time, `t ∈ [0,1]` |
| `v_theta` | learned velocity field |
| `F_v` | SeeCoder visual-context features |
| `F_d` | DiffAE damage-aware features |
| `F_R` | restored/refined feature map entering SRA |
| `A_s` | SRA attention/gating map |
| `R(F_R)` | SRA residual correction |
| `F_rect` | SRA-rectified feature |
| `P` | LCR shared projection head |
| `L` | loss |
| `lambda` | loss weight |
| `(*)` | elementwise (Hadamard) product |
| `sigma` | logistic sigmoid |

## 2. The restoration inverse problem

Image restoration assumes an observation model

```
y = D(x) + n
```

with `D` an unknown degradation operator (motion blur, rain, haze, low light,
noise, downsampling) and `n` sensor noise. Recovering `x` from `y` is
ill-posed: many `x` explain the same `y`. A learned restorer approximates the
inverse conditioned on data:

```
x_hat = F(y; theta) ≈ x
```

For GoPro, `D` is motion blur from camera/object motion during exposure.

## 3. MDTA — Multi-DConv Head Transposed Attention

For `X ∈ R^(B×C×H×W)`:

```
[Q, K, V] = DWConv_3x3( Conv_1x1(X) )        each (B, C, H, W)
reshape   Q, K, V -> (B, h, d, HW),   C = h·d
Q_hat = Q / ||Q||_2   along the HW axis
K_hat = K / ||K||_2   along the HW axis
A     = Softmax( (Q_hat · K_hat^T) * alpha )         (B, h, d, d)
Out   = Conv_1x1( reshape(A · V) )                   (B, C, H, W)
```

The contraction `Q_hat · K_hat^T` sums over the **spatial** index, so the
attention matrix is `d × d` — a *channel* affinity, not a pixel affinity.

### 3.1 Complexity — and the bug this replaces

| Formulation | Attention matrix | Cost |
|---|---|---|
| Naive spatial self-attention | `HW × HW` | `O((HW)² · C)` |
| MDTA channel attention | `d × d` | `O(HW · C²)` |

At `H = W = 256`, `C = 48`:

```
HW      = 65,536
(HW)²   = 4,294,967,296        elements  -> ~17 GB per head in fp32
C²      = 2,304                elements
```

An earlier revision of this project computed `Q^T · K` instead of `Q · K^T`,
which contracts the *channel* index and leaves an `HW × HW` matrix. That
attempted a ~34 GB allocation and aborted with
`RuntimeError: DefaultCPUAllocator: not enough memory`. **This was an
algorithmic error, not a hardware limitation.** The corrected order is in
`models/common/attention.py`, and `tests/test_attention.py` asserts via a
`torch.matmul` interception that no intermediate square matrix ever exceeds the
per-head channel bound.

## 4. GDFN — Gated-Dconv Feed-Forward Network

With `hidden = ⌊dim · 2.66⌋`:

```
X1      = Conv_1x1(X)                    -> 2·hidden channels
X2      = DWConv_3x3(X1)
Xa, Xb  = split(X2)                      -> hidden each
G       = GELU(Xa) (*) Xb
Y       = Conv_1x1(G)                    -> dim
```

The multiplicative gate lets the network suppress uninformative activations
before the output projection.

## 5. Transformer block

```
X' = X  + MDTA( LN(X)  )
Y  = X' + GDFN( LN(X') )
```

`LN` is channel-wise LayerNorm, either `BiasFree` (`x / sqrt(var + eps) · w`)
or `WithBias` (`(x - mu)/sqrt(var + eps) · w + b`).

## 6. Restormer backbone

```
F0        = OverlapPatchEmbed(y)                      (B, 48,  H,   W )
F1        = Stage_1(F0)                               (B, 48,  H,   W )
F2        = Stage_2(Down(F1))                         (B, 96,  H/2, W/2)
F3        = Stage_3(Down(F2))                         (B, 192, H/4, W/4)
Z         = Bottleneck(Down(F3))                      (B, 384, H/8, W/8)
D3        = Stage_3'( Conv_1x1[ Up(Z)  ; F3 ] )       (B, 192, H/4, W/4)
D2        = Stage_2'( Conv_1x1[ Up(D3) ; F2 ] )       (B, 96,  H/2, W/2)
D1        = Stage_1'(          [ Up(D2) ; F1 ] )      (B, 96,  H,   W )
F_R       = Refinement(D1)                            (B, 96,  H,   W )
x_hat     = Conv_3x3(F_R) + y                         (B, 3,   H,   W )
```

`Down` = `Conv_3x3(C→C/2)` then `PixelUnshuffle(2)`; `Up` = `Conv_3x3(C→2C)`
then `PixelShuffle(2)`. Decoder level 1 deliberately keeps the concatenated
`2·dim` width (no `1×1` reduction), following the original architecture; the
refinement stage therefore also runs at `2·dim`. The final `+ y` is a global
residual, so the network predicts a *correction* rather than the image.

## 7. SeeCoder — visual context

```
F_v = E_v(y; theta_v)
```

A convolutional pyramid (depthwise-separable blocks, strides 2/4/8) with one
MDTA block at the deepest level:

```
features : (B, 48, H/2, W/2), (B, 96, H/4, W/4), (B, 192, H/8, W/8)
context  : (B, 256, H/8, W/8)
```

## 8. DiffAE — damage-aware semantic encoder

```
F_d = E_d(y; theta_d)   ->  spatial (B, 256, H/8, W/8),  semantic (B, 256)
```

Interpreted as a *learned damage-aware representation*. No disentanglement
guarantee is claimed or enforced.

## 9. LCR — Latent Consistency Refinement (primary novelty)

Fusion:

```
V     = Conv_1x1(F_v),   Dm = Conv_1x1(F_d)     -> hidden_dim each
Zf    = Conv_1x1([ V ; Dm ])
Zf    = Zf + MDTA( LN(Zf) )
Zf    = RefinementBlocks(Zf)
z0    = Conv_1x1(Zf)                            -> (B, 384, H/8, W/8)
```

Consistency objective, with a **shared** projection `P` and L2 normalization:

```
P(z)   = normalize( MLP( GAP(z) ) )                (B, proj_dim), ||P(z)|| = 1
L_LCR  = || P(z_pred) - P( stopgrad(z_ref) ) ||_2^2
```

`z_ref` is the backbone latent of the clean target (training only). Normalizing
prevents the degenerate solution of shrinking latent norms toward zero;
`stopgrad` keeps the clean latent a fixed target.

## 10. Rectified Flow — latent transport

Training:

```
t     ~ U(0,1)
z_t   = (1-t)·z0 + t·z1
v*    = z1 - z0
L_flow = || v_theta(z_t, t) - (z1 - z0) ||_2^2
```

Inference (`z1` unavailable) — solve the ODE:

```
dz/dt = v_theta(z, t),      z(0) = z0
```

Euler, `dt = 1/N`, `N = 10..20`:

```
z_{k+1} = z_k + dt · v_theta(z_k, k·dt)
```

An optional Heun (2nd-order) solver is available. Because the target velocity
is *constant* along the straight path `z0 → z1`, few steps suffice. `v_theta`
ends in a zero-initialized convolution, so at initialization `v_theta ≡ 0` and
the ODE is the identity map.

## 11. SRA — Self-Rectifying Attention

```
A_s     = sigma( Conv_1x1( MDTA( LN(F_R) ) ) )          in (0,1)
R(F_R)  = Conv_1x1( GELU( DWConv_3x3( Conv_1x1( LN(F_R) ) ) ) )
F_rect  = F_R + gamma · ( A_s (*) R(F_R) )
```

Both the correction and the gate that admits it derive from `F_R` alone — hence
*self*-rectifying. `gamma` is a learnable scalar initialized to **zero**, so
SRA is an exact no-op at initialization. Attention is channel-wise (MDTA); a
spatial map here would be at full decoder resolution and would reintroduce the
`HW × HW` blow-up of §3.1.

## 12. Full SRRT

```
F_v  = E_v(y)
F_d  = E_d(y)
z0   = L(F_v, F_d)
z1_h = ODE_solve( v_theta, z0 )                  (inference)
Z'   = Z + gamma_latent · Conv_1x1(z1_h)         injected at the bottleneck
...decoder + refinement as in §6...
F_rect = SRA(F_R)
x_hat  = Conv_3x3(F_rect) + y
```

`gamma_latent` is zero-initialized, so at initialization SRRT reduces exactly
to the Restormer baseline of §6 — verified numerically in
`tests/test_srrt.py::test_baseline_srrt_is_numerically_identical_to_restormer`.

## 13. Combined objective

```
L_total = lambda_1 · L_L1
        + lambda_p · L_perceptual
        + lambda_f · L_flow
        + lambda_c · L_LCR
```

with

```
L_L1         = mean( |x_hat - x| )
L_perceptual = sum_l w_l · mean( | phi_l(x_hat) - phi_l(x) | )      (frozen VGG19)
L_flow       = §10
L_LCR        = §9
```

Initial weights: `lambda_1 = 1.0`, `lambda_p = 0.1`, `lambda_f = 0.1`,
`lambda_c = 0.1` — all YAML-configurable and **not** tuned or validated. The
flow and LCR terms are added only when their modules are enabled.

`phi_l` are VGG19 features at `relu1_2, relu2_2, relu3_4, relu4_4`. LPIPS is
deliberately excluded from training so it remains an independent evaluation
metric.

## 14. Evaluation metrics

```
PSNR = 10 · log10( MAX_I² / MSE ),    MAX_I = 1,   MSE = mean((x_hat - x)²)
```

PSNR ↑ better, SSIM ↑ better, LPIPS ↓ better.
