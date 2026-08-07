# burn-kda - Kimi Delta Attention for Burn

[![CI](https://github.com/sehaxe/burn-kda/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-kda/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-kda)](https://crates.io/crates/burn-kda)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.21-orange.svg)](https://burn.dev)

Complete implementation of Kimi Delta Attention (KDA) from
[Kimi Linear](https://arxiv.org/abs/2510.26692) and
[Kimi K3](https://arxiv.org/abs/2607.24653) (§2.1.1, Eqs 1-6).

KDA extends the gated delta rule with a **channel-wise forget gate** and a
data-dependent write strength. The core recurrence (Eq 1):

```text
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
o_t = S_t^T q_t
```

## What is implemented

- **Data-dependent write strength** (K3 Eq 2): `beta_t^h = Sigmoid(W_beta^h x_t)`,
  per-head and per-token - not a global scalar.
- **Channel-wise decay** from a low-rank logit
  `z_t = W_alpha^down(W_alpha^up x_t) + b_alpha` with rank = head_dim (both
  papers) and per-head log-scale `A_h`, with two mappings:
  - `DecayFn::Softplus` (Kimi Linear): `g = -exp(A_h) Softplus(z)`, `alpha in (0, 1)`
  - `DecayFn::Sigmoid` (Kimi K3): `g = g_min Sigmoid(exp(A_h) z)`, fixed `g_min = -5`,
    `alpha in (e^-5, 1)` - bounded, numerically safe
- **q/k = L2Norm(Swish(ShortConv(Wx)))**, **v = Swish(ShortConv(W_v x))** (K3 Eq 2).
- **Output gate** (K3 Eq 6): `y = W_o [Sigmoid(W_g x) * RMSNorm(o_t)]`, with
  `GateMode::FullRank` (K3) or `GateMode::LowRank` (Kimi Linear Eq 10).
- **Training**: chunked WY form (identical algebra to GDN-2 under the mapping
  `b = beta`, `g = log(alpha)`, `w = beta`).
- **Decoding**: exact per-token recurrence (Eq 1).
- **Fused CUDA path** (`feature = "cuda"`): the GDN-2 chunked kernels are
  reused through the same WY mapping (no value-gate trick: `U` carries the
  `beta` write strength as the value gate).

> **Training:** `forward_train_fused` runs the whole chunked WY recurrence as
> ONE autodiff node through burn-gdn2 0.7's fused op (exact matrix-level
> backward; the fused CUDA kernels run the forward on `CudaBare`, the
> backward never re-runs the forward). `forward_train` is the plain tensor
> path for any backend.

## Install

```bash
cargo add burn-kda
# fused CUDA kernels:
cargo add burn-kda --features cuda
```

## Quick start

```rust
use burn_kda::{DecayFn, GateMode, KdaConfig, KdaModule};

let cfg = KdaConfig {
    hidden_size: 128,
    num_heads: 4,
    head_dim: 32,
    decay_fn: DecayFn::Sigmoid, // or Softplus
    gate: GateMode::FullRank,   // or LowRank
    chunk_size: 16,             // stable for g_min = -5 in f32
    ..Default::default()
};
let layer = KdaModule::new(&cfg, 0.0, &device);

let out_train = layer.forward_train(x);                 // chunked (WY) path
let (out_decode, state) = (0..T).fold((Tensor::zeros([..]), None), |..|);
```

## Numerical stability

With `g_min = -5` a naive chunk of 64 accumulates log-decay down to -320, so
`1/Gamma = exp(320)` overflows f32 (the same reason K3 uses 16-token
secondary tiles). Two layers of protection:

- The tensor chunk path (burn-gdn2 0.5.1) computes decay tile-locally in log
  space with inter-tile weights `exp(G_p - G_q) <= 1` and solves the WY
  system per 16-token tile, so **any chunk size is exact** — `chunk_size: 64`
  (the K3 paper setting) is supported.
- The fused CUDA kernel reuses the GDN-2 kernel, which is only stable up to
  `chunk_size = 16`; larger chunks fall back to the tensor path
  automatically.

## Tests

- 10 unit tests (decay bounds for both mappings, data-dependent beta,
  chunk == decode for both mappings and for chunk 64, gate modes, shapes).
- 3 CUDA tests (fused == tensor chunk, fused == decode for both mappings).

> Papers: Kimi Linear (Moonshot, 2025); Kimi K3 (Moonshot, 2026).

## Performance

Attention core only (the chunked delta-rule, no projections/conv/gate), same
GPU (RTX 5060 Ti), warmup + averaged. burn side: `tests/bench_cuda.rs`
(fused kernels forward, fused autodiff op train); torch side: `bench_kda.py`
(fla `chunk_kda`, the K3 Triton path that FlashKDA accelerates).

| config | burn fwd | fla fwd | burn module train | fla train |
|--------|----------|---------|-------------------|-----------|
| d=512, h=4, hk=64, T=1024 | 0.10 ms | 1.0 ms | 0.36 ms | 2.8 ms |
| d=1024, h=8, hk=64, T=2048 | 0.05 ms | 1.0 ms | 0.43 ms | 4.7 ms |
| d=2048, h=8, hk=128, T=4096 | 0.05 ms | 1.1 ms | 0.42 ms | 4.6 ms |
| d=4096, h=8, hk=128, T=8192 | 0.05 ms | 2.1 ms | 0.43 ms | 8.2 ms |

Forward: **15-40x faster than the Triton reference** (2 fused launches per
chunk vs Triton's per-chunk kernel set; FlashKDA's CUTLASS path was not
buildable here - it needs a CUDA 12.8 toolkit matching torch cu128).

Training: with burn-gdn2 0.7.5 the fused chunked backward (BK2 inter BPTT +
BK1 intra adjoint) runs on 2 kernels, and the fused op path skips the
per-chunk scratch, so the module train (projections included) is **0.36-0.43
ms** — **7-24x faster than the Triton reference's raw attention-core train**.
