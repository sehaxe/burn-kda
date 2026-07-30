//! # burn-kda — Kimi Delta Attention for Burn
//!
//! | Component | What |
//! |-----------|------|
//! | `ChannelDecay` | Standalone per-channel learnable decay `[H, D]` in `[min_d, 1]` |
//! | `kda_step` | One delta-rule step: decay → predict → correct → read |
//! | `KdaModule` | GDN2 + per-channel lower-bounded decay — drop-in upgrade |
//!
//! Kimi K3 (Moonshot, 2026): 69 KDA layers with per-channel lower-bounded decay.
use burn::module::Module;
use burn::nn::Initializer;
use burn::tensor::{activation, backend::Backend, Tensor};
use burn_gdn2::{GatedDeltaNet2, Gdn2Config, Gdn2Mode};

// ─── Standalone Channel Decay ────────────────────────────────────────

#[derive(Module, Debug)]
pub struct ChannelDecay<B: Backend> {
    pub raw: burn::module::Param<Tensor<B, 2>>,
    pub min_decay: f64,
}

impl<B: Backend> ChannelDecay<B> {
    pub fn new(n_heads: usize, head_dim: usize, min_decay: f64, device: &B::Device) -> Self {
        Self {
            raw: Initializer::Constant { value: 2.0 }.init([n_heads, head_dim], device),
            min_decay,
        }
    }

    pub fn forward(&self) -> Tensor<B, 2> {
        let raw = activation::sigmoid(self.raw.val());
        let alpha = 1.0 - self.min_decay;
        raw.mul_scalar(alpha).add_scalar(self.min_decay)
    }
}

// ─── KDA step ─────────────────────────────────────────────────────────

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

// ─── KdaModule: GDN2 with per-channel lower-bounded decay ─────────────

/// Drop-in upgrade for GatedDeltaNet2 with per-channel lower-bounded decay.
///
/// Creates a GDN2 with `min_decay` set, adding learned per-channel factors
/// that guarantee a minimum decay floor. From Kimi K3.
#[derive(Module, Debug)]
pub struct KdaModule<B: Backend> {
    pub gdn2: GatedDeltaNet2<B>,
}

impl<B: Backend> KdaModule<B> {
    pub fn new(cfg: &Gdn2Config, min_decay: f64, device: &B::Device) -> Self {
        let mut cfg = cfg.clone();
        cfg.min_decay = Some(min_decay);
        Self {
            gdn2: GatedDeltaNet2::new(&cfg, device),
        }
    }

    pub fn forward_train(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.gdn2.forward_train(x)
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        state: &mut Option<Tensor<B, 4>>,
        update_state: bool,
    ) -> Tensor<B, 3> {
        self.gdn2.forward(x, state, update_state)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

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
        let v: Vec<f32> = d
            .forward()
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(v.iter().all(|x| (0.89..=1.01).contains(x)));
    }
    #[test]
    fn kda_step_shapes() {
        let s = Tensor::<B, 4>::zeros([2, 4, 32, 64], &dev());
        let d = Tensor::<B, 2>::ones([4, 32], &dev()).mul_scalar(0.95);
        let q = Tensor::<B, 3>::random([2, 4, 32], Distribution::Default, &dev());
        let k = Tensor::<B, 3>::random([2, 4, 32], Distribution::Default, &dev());
        let v = Tensor::<B, 3>::random([2, 4, 64], Distribution::Default, &dev());
        let (ns, o) = kda_step(s, d, q, k, v, 0.5);
        assert_eq!(ns.dims(), [2, 4, 32, 64]);
        assert_eq!(o.dims(), [2, 4, 64]);
    }
    #[test]
    fn kda_module_forward() {
        let cfg = Gdn2Config {
            hidden_size: 64,
            num_heads: 2,
            head_dim: 32,
            ..Default::default()
        };
        let km = KdaModule::new(&cfg, 0.9, &dev());
        let x = Tensor::<B, 3>::random([1, 4, 64], Distribution::Default, &dev());
        assert_eq!(km.forward_train(x).dims(), [1, 4, 64]);
    }
}
