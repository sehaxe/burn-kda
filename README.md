# burn-kda — Kimi Delta Attention for Burn

[![CI](https://github.com/sehaxe/burn-kda/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-kda/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-kda)](https://crates.io/crates/burn-kda)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.21-orange.svg)](https://burn.dev)

Per-channel lower-bounded decay with delta-rule state update (Kimi K3).
Each memory channel has its own learnable decay factor. Delta-rule update
writes new information as the difference between expected and actual value.

> Paper: Kimi K3 (Moonshot, 2026). 69 KDA layers + 24 Gated MLA layers.

## Install

```bash
cargo add burn-kda
```

## Quick start

```rust
use burn_kda::{ChannelDecay, kda_step};

let decay = ChannelDecay::new(8, 64, 0.9, &device); // 8 heads, 64-dim, min 0.9
let d = decay.forward(); // [8, 64] in [0.9, 1.0]

let (new_state, output) = kda_step(state, d, q, k, v, 0.5);
```

## API

| Export | What |
|--------|------|
| `ChannelDecay` | Learnable per-channel decay `[H, D]` in `[min_decay, 1]` |
| `kda_step` | One delta-rule step: decay → predict → correct → read |

## License

AGPL-3.0. See [LICENSE](LICENSE).
