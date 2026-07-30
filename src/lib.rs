//! # burn-kda — Kimi Delta Attention for Burn
//!
//! Per-channel lower-bounded decay with delta-rule state update (Kimi K3).
//! Each memory channel has its own learnable decay factor, bounded from below
//! for numerical stability. Delta-rule update writes new information as the
//! difference between expected and actual value.
//!
//! | Paper | What |
//! |-------|------|
//! | Kimi K3 (Moonshot, 2026) | Per-channel decay, lower-bounded, delta-rule |
//!
//! ```text
//! decay = min_decay + (1-min_decay) * sigmoid(raw_decay)  [H, D]
//! S_t = S_{t-1} * decay_t                    state decay
//! v_hat = S_t @ k_t                          predict
//! e_t = beta_t * (v_t - v_hat)               delta error
//! S_t += k_t @ e_t^T                         update state
//! o_t = q_t @ S_t                            read output
//! ```
use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::tensor::{activation, backend::Backend, Tensor};

/// Per-channel lower-bounded decay for recurrent state.
///
/// `raw_decay`: `[H, D]` — learnable per-channel logits
/// `min_decay`: minimum decay (e.g., 0.9 for KDA, 0.5 for conservative)
///
/// Returns `[H, D]` — bounded decay factors in `[min_decay, 1.0]`.
///
/// From Kimi K3: lower-bounded decay prevents catastrophic forgetting
/// while still allowing selective channel erasure.
#[derive(Module, Debug)]
pub struct ChannelDecay<B: Backend> {
    pub raw: Param<Tensor<B, 2>>,
    pub min_decay: f64,
}

impl<B: Backend> ChannelDecay<B> {
    pub fn new(n_heads: usize, head_dim: usize, min_decay: f64, device: &B::Device) -> Self {
        Self {
            raw: Initializer::Constant { value: 2.0 }.init([n_heads, head_dim], device),
            min_decay,
        }
    }

    /// Bounded decay: `min_decay + (1-min_decay) * sigmoid(raw)`.
    pub fn forward(&self) -> Tensor<B, 2> {
        let raw = activation::sigmoid(self.raw.val());
        let alpha = 1.0 - self.min_decay;
        raw.mul_scalar(alpha).add_scalar(self.min_decay)
    }
}

/// Kimi Delta Attention step: state update with per-channel decay.
///
/// Applies one step of the delta rule:
/// - Decay state with per-channel factors
/// - Compute prediction error
/// - Update state with delta correction
/// - Read output
pub fn kda_step<B: Backend>(
    state: Tensor<B, 4>,
    decay: Tensor<B, 2>,
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    beta: f64,
) -> (Tensor<B, 4>, Tensor<B, 3>) {
    let [b, h, dk, dv] = state.dims();

    let state = state.mul(decay.clone().reshape([1, h, dk, 1]));
    let v_hat = state
        .clone()
        .swap_dims(2, 3)
        .matmul(k.clone().reshape([b, h, dk, 1]))
        .reshape([b, h, dv]);
    let delta = v.clone().sub(v_hat).mul_scalar(beta);
    let state = state + k.reshape([b, h, dk, 1]).mul(delta.reshape([b, h, 1, dv]));
    let out = q
        .reshape([b * h, 1, dk])
        .matmul(state.clone().reshape([b * h, dk, dv]))
        .reshape([b, h, dv]);
    (state, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use burn_ndarray::{NdArray, NdArrayDevice};
    type B = NdArray;
    fn dev() -> NdArrayDevice {
        NdArrayDevice::default()
    }

    #[test]
    fn channel_decay_range() {
        let d = ChannelDecay::<B>::new(4, 64, 0.9, &dev());
        let vals: Vec<f32> = d
            .forward()
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for v in vals {
            assert!(v >= 0.89 && v <= 1.01, "decay {v} should be in [0.9, 1.0]");
        }
    }
    #[test]
    fn kda_step_shapes() {
        let s = Tensor::<B, 4>::zeros([2, 4, 32, 64], &dev());
        let d = Tensor::<B, 2>::ones([4, 32], &dev()).mul_scalar(0.95);
        let q = Tensor::<B, 3>::random([2, 4, 32], Distribution::Default, &dev());
        let k = Tensor::<B, 3>::random([2, 4, 32], Distribution::Default, &dev());
        let v = Tensor::<B, 3>::random([2, 4, 64], Distribution::Default, &dev());
        let (new_s, out) = kda_step(s, d, q, k, v, 0.5);
        assert_eq!(new_s.dims(), [2, 4, 32, 64]);
        assert_eq!(out.dims(), [2, 4, 64]);
    }
}
