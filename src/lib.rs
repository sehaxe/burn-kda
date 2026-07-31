//! # burn-kda - Kimi Delta Attention for Burn
//!
//! Full implementation of the KDA mixer from [Kimi K3](https://arxiv.org/abs/2607.24653)
//! (§2.1.1, Eqs 1-5):
//!
//! - Eq 1: `S_t = (I - beta*k*k^T) Diag(alpha_t) S_{t-1} + beta*k*v^T`,
//!   `o_t = S_t^T q_t` - delta rule with channel-wise lower-bounded decay
//! - Eq 2: data-dependent low-rank logit `z_t = W^down(W^up(x_t)) + b_alpha`
//! - Eq 5: `g_t = g_min * sigmoid(e^{A_h} * z_t)`, `alpha_t = exp(g_t)`
//!   with fixed `g_min = -5` (alpha in `(e^{g_min}, 1)`) and per-head
//!   log-scale `A_h`
//!
//! `KdaModule` is a complete KDA layer: q/k/v projections with short conv,
//! L2-normalized keys (K3 block design), the data-dependent decay above,
//! scalar write strength `beta in (0,1)`, and the K3 output gate
//! `Sigmoid(W_g x) * RMSNorm(o)`. Training uses the chunked delta-rule form
//! (the same WY chunk algebra as GDN-2 with b = w = beta); decoding uses the
//! exact per-token recurrence (Eq 1).
use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::tensor::{activation, backend::Backend, Tensor};
use burn_gdn2::{chunk_wy_forward, l2_normalize, short_conv_1d, Gdn2Config};

/// Paper's fixed log-space decay floor (Eq 5: `g_min = -5`, alpha > e^-5).
pub const G_MIN: f64 = -5.0;

// ─── Data-dependent decay (Eq 2/5) ────────────────────────────────────

/// Data-dependent channel-wise decay (K3 report Eq 2/5).
///
/// ```text
/// z_t    = W^down(W^up(x_t)) + b_alpha            (Eq 2, low-rank logit)
/// g_t    = g_min * sigmoid(e^{A_h} * z_t)          (Eq 5, per-head log scale)
/// alpha_t = exp(g_t)  in (e^{g_min}, 1)
/// ```
#[derive(Module, Debug)]
pub struct KdaDecay<B: Backend> {
    pub w_up: Linear<B>,
    pub w_down: Linear<B>,
    pub b_alpha: Param<Tensor<B, 1>>,
    /// Per-head log-scale `A_h` (Eq 5), shape `[n_heads, 1]`.
    pub a_log: Param<Tensor<B, 2>>,
    #[module(skip)]
    pub g_min: f64,
}

impl<B: Backend> KdaDecay<B> {
    /// `rank`: low-rank logit dimension `r` (Eq 2).
    /// `g_min`: log-space decay floor; the paper fixes it at -5; pass
    /// `ln(min_decay)` to reinterpret a decay-space floor (aria convention).
    pub fn new(
        d_model: usize,
        n_heads: usize,
        head_dim: usize,
        rank: usize,
        g_min: f64,
        device: &B::Device,
    ) -> Self {
        let init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        Self {
            w_up: LinearConfig::new(d_model, rank)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            w_down: LinearConfig::new(rank, n_heads * head_dim)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            b_alpha: Param::from_tensor(Tensor::zeros([n_heads * head_dim], device)),
            a_log: Param::from_tensor(Tensor::zeros([n_heads, 1], device)),
            g_min,
        }
    }

    /// `x`: `[B, T, D]` hidden states. Returns `alpha [B, T, H, HD]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let z = self.w_down.forward(self.w_up.forward(x));
        let [b2, t2, hd2] = z.dims();
        let z = z.add(self.b_alpha.val().clone().reshape([1, 1, hd2]));
        let n_heads = self.a_log.val().dims()[0];
        let head_dim = hd2 / n_heads;
        let z_h = z.reshape([b2, t2, n_heads, head_dim]);
        let scaled = z_h.mul(self.a_log.val().clone().reshape([1, 1, n_heads, 1]).exp());
        let g = activation::sigmoid(scaled).mul_scalar(self.g_min as f32);
        g.exp()
    }
}

// ─── Exact single-step recurrence (Eq 1) ──────────────────────────────

/// One KDA step (Eq 1): decay -> erase -> write -> read.
///
/// `state`: `[B, H, DK, DV]`, `decay`: `[H, DK]`, `q/k`: `[B, H, DK]`,
/// `v`: `[B, H, DV]`, `beta`: scalar write strength in `(0, 1)`.
///
/// Returns `(state, out [B, H, DV])`.
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

// ─── Full KDA layer ───────────────────────────────────────────────────

/// Kimi Delta Attention layer (K3 report §2.1.1): delta rule with
/// data-dependent lower-bounded decay.
///
/// Block design (K3): `q,k = L2Norm(SiLU(ShortConv(Linear(x))))`,
/// `v = SiLU(ShortConv(Linear(x)))`; decay from Eq 2/5; scalar `beta`;
/// output gate `Sigmoid(W_g x) * RMSNorm(o)` before `o_proj`.
#[derive(Module, Debug)]
pub struct KdaModule<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub q_conv_w: Option<Param<Tensor<B, 2>>>,
    pub k_conv_w: Option<Param<Tensor<B, 2>>>,
    pub v_conv_w: Option<Param<Tensor<B, 2>>>,
    pub decay: KdaDecay<B>,
    /// Scalar write strength, sigmoid-parameterized into `(0, 1)` (Eq 1 beta).
    pub beta: Param<Tensor<B, 1>>,
    /// Output gate `W_g` (K3: `Sigmoid(W_g x) * RMSNorm(o)`).
    pub o_gate: Linear<B>,
    pub o_norm_w: Param<Tensor<B, 1>>,
    pub o_proj: Linear<B>,
    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub n_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub n_v_heads: usize,
    #[module(skip)]
    pub v_head_dim: usize,
    #[module(skip)]
    pub use_short_conv: bool,
    #[module(skip)]
    pub chunk_size: usize,
    #[module(skip)]
    pub norm_eps: f64,
}

impl<B: Backend> KdaModule<B> {
    /// Builds from a `Gdn2Config` (drop-in for the old wrapper).
    /// `min_decay`: decay-space floor, reinterpreted as the paper's log-space
    /// `g_min = ln(min_decay)`; `min_decay <= 0` uses the paper's fixed -5.
    pub fn new(cfg: &Gdn2Config, min_decay: f64, device: &B::Device) -> Self {
        let d = cfg.hidden_size;
        let h = cfg.num_heads;
        let hk = cfg.head_dim;
        let hv = cfg.num_v_heads.unwrap_or(h);
        let v_head = (hk as f32 * cfg.expand_v) as usize;
        let init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let rank = 64.min(d);
        let g_min = if min_decay > 0.0 {
            min_decay.ln()
        } else {
            G_MIN
        };
        let conv = |dim: usize| -> Option<Param<Tensor<B, 2>>> {
            cfg.use_short_conv.then(|| {
                Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                }
                .init([dim, 4], device)
            })
        };
        Self {
            q_proj: LinearConfig::new(d, h * hk)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            k_proj: LinearConfig::new(d, h * hk)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            v_proj: LinearConfig::new(d, hv * v_head)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            q_conv_w: conv(h * hk),
            k_conv_w: conv(h * hk),
            v_conv_w: conv(hv * v_head),
            decay: KdaDecay::new(d, h, hk, rank, g_min, device),
            beta: Param::from_tensor(Tensor::ones([1], device).mul_scalar(1.0)),
            o_gate: LinearConfig::new(d, hv * v_head)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            o_norm_w: Initializer::Ones.init([v_head], device),
            o_proj: LinearConfig::new(hv * v_head, d)
                .with_bias(false)
                .with_initializer(init)
                .init(device),
            d_model: d,
            n_heads: h,
            head_dim: hk,
            n_v_heads: hv,
            v_head_dim: v_head,
            use_short_conv: cfg.use_short_conv,
            chunk_size: cfg.chunk_size,
            norm_eps: cfg.norm_eps,
        }
    }

    /// Projections + decay (K3 block design). Returns `(q, k, v, log_decay,
    /// beta, gate_signal)` with heads expanded for grouped value attention.
    #[allow(clippy::type_complexity)]
    fn project(
        &self,
        x: Tensor<B, 3>,
    ) -> (
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
    ) {
        let [batch, tokens, _] = x.dims();
        let h = self.n_heads;
        let hk = self.head_dim;
        let hv = self.n_v_heads;
        let vd = self.v_head_dim;
        let to_4d = |t: Tensor<B, 3>, n: usize, d: usize| -> Tensor<B, 4> {
            let [b, tt, _] = t.shape().dims::<3>();
            t.reshape([b, tt, n, d]).permute([0, 2, 1, 3])
        };

        let q_raw = self.q_proj.forward(x.clone());
        let k_raw = self.k_proj.forward(x.clone());
        let v_raw = self.v_proj.forward(x.clone());

        let q_act = if self.use_short_conv {
            short_conv_1d(q_raw, self.q_conv_w.as_ref().unwrap().val())
        } else {
            activation::silu(q_raw)
        };
        let k_act = if self.use_short_conv {
            short_conv_1d(k_raw, self.k_conv_w.as_ref().unwrap().val())
        } else {
            activation::silu(k_raw)
        };
        let v_act = if self.use_short_conv {
            short_conv_1d(v_raw, self.v_conv_w.as_ref().unwrap().val())
        } else {
            activation::silu(v_raw)
        };

        let q_norm = l2_normalize(q_act, 1e-6);
        let k_norm = l2_normalize(k_act, 1e-6);

        let alpha = self.decay.forward(x.clone()); // [B, T, H, HD]
        let [_, _, _, hd] = alpha.dims();
        let log_decay = alpha.log().permute([0, 2, 1, 3]); // [B, H, T, HD]
        let _ = hd;

        let beta = activation::sigmoid(self.beta.val()); // scalar in (0,1)
        let b_4d = beta
            .clone()
            .reshape([1, 1, 1, 1])
            .repeat(&[batch, h, tokens, hk]);

        let mut q_4d = to_4d(q_norm, h, hk);
        let mut k_4d = to_4d(k_norm, h, hk);
        let v_4d = to_4d(v_act, hv, vd);
        let mut g_4d = log_decay;

        // Repeat key-side tensors for grouped value attention (GVA)
        if hv > h {
            let rep = hv / h;
            let r = |t: Tensor<B, 4>| -> Tensor<B, 4> {
                t.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, rep, 1])
                    .reshape([batch, hv, tokens, hk])
            };
            q_4d = r(q_4d);
            k_4d = r(k_4d);
            g_4d = r(g_4d);
        }

        let gate_4d = activation::sigmoid(self.o_gate.forward(x))
            .reshape([batch, tokens, hv, vd])
            .permute([0, 2, 1, 3]);

        (q_4d, k_4d, v_4d, g_4d, b_4d, gate_4d)
    }

    fn output(&self, attn_out: Tensor<B, 4>, gate: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, hv, t, vd] = attn_out.dims();
        let rms = attn_out
            .clone()
            .powf_scalar(2.0)
            .mean_dim(3)
            .add_scalar(self.norm_eps)
            .sqrt();
        let normed = attn_out / rms;
        let gated = normed
            .mul(gate)
            .mul(self.o_norm_w.val().reshape([1, 1, 1, vd]));
        self.o_proj
            .forward(gated.permute([0, 2, 1, 3]).reshape([b, t, hv * vd]))
    }

    /// Training forward: chunked delta-rule (WY algebra, b = w = beta).
    pub fn forward_train(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _t, _] = x.dims();
        let (q, k, v, g, b_gate, gate) = self.project(x);
        let [_, hv, _, _] = v.dims();
        let state = Tensor::zeros([batch, hv, self.head_dim, self.v_head_dim], &q.device());
        let (out, _) = chunk_wy_forward(
            q,
            k,
            v,
            g,
            b_gate.clone(),
            b_gate,
            state,
            1.0,
            self.chunk_size,
        );
        self.output(out, gate)
    }

    /// Decode forward: exact per-token recurrence (Eq 1).
    ///
    /// `update_state=true`: state is decayed/updated per token (autoregressive
    /// decoding). `update_state=false`: read-only prefill over the state.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        state: &mut Option<Tensor<B, 4>>,
        update_state: bool,
    ) -> Tensor<B, 3> {
        let [batch, tokens, _] = x.dims();
        let (q, k, v, g, b, gate) = self.project(x.clone());
        let [_, hv, _, _] = v.dims();
        let dev = q.device();
        let s = state
            .take()
            .unwrap_or_else(|| Tensor::zeros([batch, hv, self.head_dim, self.v_head_dim], &dev));

        let beta = activation::sigmoid(self.beta.val()).reshape([1, 1, 1, 1]);
        let _ = b;
        let (out_4d, new_state) = if update_state {
            let mut s = s;
            let mut outs = Vec::with_capacity(tokens);
            for t in 0..tokens {
                let q_t = q.clone().slice_dim(2, t..t + 1);
                let k_t = k.clone().slice_dim(2, t..t + 1);
                let v_t = v.clone().slice_dim(2, t..t + 1);
                let d_t = g.clone().slice_dim(2, t..t + 1).exp();
                // S <- Diag(alpha_t) S  (Eq 1)
                s = s * d_t.swap_dims(2, 3);
                // erase: S <- S - beta k (k^T S)
                let erased = (s.clone() * k_t.clone().swap_dims(2, 3))
                    .sum_dim(2)
                    .mul(beta.clone());
                s = s - k_t.clone().swap_dims(2, 3) * erased;
                // write: S <- S + beta k v^T
                s = s + k_t.swap_dims(2, 3) * v_t.mul(beta.clone());
                // read: o = q^T S
                let out = (s.clone() * q_t.swap_dims(2, 3)).sum_dim(2);
                outs.push(out);
            }
            (Tensor::cat(outs, 2), s)
        } else {
            let out = q.matmul(s.clone()).permute([0, 2, 1, 3]);
            (out, s)
        };
        *state = Some(new_state);
        self.output(out_4d, gate)
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
    fn kda_decay_bounds() {
        // Eq 5: alpha = exp(g_min*sigmoid(...)) in (e^{g_min}, 1)
        let dec = KdaDecay::<B>::new(32, 4, 16, 16, G_MIN, &dev());
        let x = Tensor::<B, 3>::random([2, 8, 32], Distribution::Default, &dev());
        let alpha = dec.forward(x);
        let vals: Vec<f32> = alpha
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let lo = G_MIN.exp() as f32;
        assert!(
            vals.iter().all(|&v| v >= lo - 1e-6 && v <= 1.0 + 1e-6),
            "alpha out of (e^g_min, 1): lo={lo}, sample {:?}",
            &vals[..8]
        );
    }

    #[test]
    fn kda_decay_is_data_dependent() {
        // Eq 2: the logit is a function of x - two different inputs give
        // different decays.
        let dec = KdaDecay::<B>::new(32, 2, 16, 16, G_MIN, &dev());
        let x1 = Tensor::<B, 3>::ones([1, 4, 32], &dev());
        let x2 = Tensor::<B, 3>::ones([1, 4, 32], &dev()).mul_scalar(-2.0);
        let a1 = dec.forward(x1);
        let a2 = dec.forward(x2);
        let d: Vec<f32> = (a1 - a2)
            .abs()
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let max_d = d.iter().fold(0.0f32, |a, &x| a.max(x));
        assert!(max_d > 1e-3, "decay must depend on the input (Eq 2)");
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

    #[test]
    fn chunk_matches_decode() {
        // The chunked training path must equal the per-token recurrence.
        let cfg = Gdn2Config {
            hidden_size: 64,
            num_heads: 2,
            head_dim: 16,
            use_short_conv: false,
            ..Default::default()
        };
        let km = KdaModule::new(&cfg, 0.9, &dev());
        let x = Tensor::<B, 3>::random([1, 17, 64], Distribution::Default, &dev());
        let chunk_out = km.forward_train(x.clone());
        let mut state: Option<Tensor<B, 4>> = None;
        let decode_out = km.forward(x, &mut state, true);
        let diff: f32 = (chunk_out - decode_out)
            .powf_scalar(2.0)
            .mean()
            .into_scalar();
        assert!(diff < 1e-4, "chunk vs decode mismatch mse {diff}");
    }

    #[test]
    fn beta_stays_in_unit_interval() {
        let cfg = Gdn2Config {
            hidden_size: 32,
            num_heads: 1,
            head_dim: 16,
            use_short_conv: false,
            ..Default::default()
        };
        let km = KdaModule::<B>::new(&cfg, 0.0, &dev());
        let beta = activation::sigmoid(km.beta.val());
        let v: f32 = beta.into_scalar();
        assert!(v > 0.0 && v < 1.0, "beta must be in (0,1), got {v}");
    }
}
