//! Fused CUDA KDA path must match the tensor chunk path (same WY inputs).
#![cfg(feature = "cuda")]
use burn::tensor::{Distribution, Tensor};
use burn_kda::{DecayFn, GateMode, KdaConfig, KdaModule};

fn cfg() -> KdaConfig {
    KdaConfig {
        hidden_size: 128,
        num_heads: 4,
        head_dim: 32,
        use_short_conv: false,
        decay_fn: DecayFn::Sigmoid,
        gate: GateMode::FullRank,
        chunk_size: 16,
        ..Default::default()
    }
}

#[test]
fn fused_matches_decode() {
    type Bare = burn_kda::fused::cuda::CudaBare;
    let dev = Default::default();
    let km = KdaModule::<Bare>::new(&cfg(), 0.0, &dev);
    let x = Tensor::<Bare, 3>::random([1, 64, 128], Distribution::Normal(0.0, 1.0), &dev);
    let fused = km.forward_train(x.clone());
    let mut state = None;
    let decode = km.forward(x, &mut state, true);
    let dmax: f32 = (fused - decode)
        .abs()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|bb| f32::from_le_bytes(bb.try_into().unwrap()))
        .fold(0.0f32, f32::max);
    println!("fused vs decode max_diff {dmax:.3e}");
    assert!(dmax < 1e-3, "fused vs decode mismatch max_diff {dmax:.3e}");
}

#[test]
fn fused_matches_decode_softplus() {
    type Bare = burn_kda::fused::cuda::CudaBare;
    let dev = Default::default();
    let c = KdaConfig {
        decay_fn: DecayFn::Softplus,
        ..cfg()
    };
    let km = KdaModule::<Bare>::new(&c, 0.0, &dev);
    let x = Tensor::<Bare, 3>::random([1, 64, 128], Distribution::Normal(0.0, 1.0), &dev);
    let fused = km.forward_train(x.clone());
    let mut state = None;
    let decode = km.forward(x, &mut state, true);
    let dmax: f32 = (fused - decode)
        .abs()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|bb| f32::from_le_bytes(bb.try_into().unwrap()))
        .fold(0.0f32, f32::max);
    println!("softplus fused vs decode max_diff {dmax:.3e}");
    assert!(
        dmax < 1e-3,
        "softplus fused vs decode mismatch max_diff {dmax:.3e}"
    );
}

#[test]
fn fused_matches_tensor_chunk() {
    type Bare = burn_kda::fused::cuda::CudaBare;
    let dev = Default::default();
    let km = KdaModule::<Bare>::new(&cfg(), 0.0, &dev);
    let x = Tensor::<Bare, 3>::random([1, 32, 128], Distribution::Normal(0.0, 1.0), &dev);
    let proj = km.project_for_test(x);
    let q = proj.0;
    let k = proj.1;
    let v = proj.2;
    let log_alpha = proj.3;
    let beta = proj.4;
    let state = Tensor::<Bare, 4>::zeros([1, 4, 32, 32], &dev);

    let (fused_out, _) = burn_kda::fused::cuda::kda_fused_chunk::<Bare>(
        q.clone(),
        k.clone(),
        v.clone(),
        log_alpha.clone(),
        beta.clone(),
        beta.clone(),
        state.clone(),
        16,
    )
    .expect("fused path must apply on CudaBare");

    let (tensor_out, _) =
        burn_gdn2::chunk_wy_forward(q, k, v, log_alpha, beta.clone(), beta, state, 1.0, 16);

    let dmax: f32 = (fused_out - tensor_out)
        .abs()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|bb| f32::from_le_bytes(bb.try_into().unwrap()))
        .fold(0.0f32, f32::max);
    println!("fused vs tensor-chunk max_diff {dmax:.3e}");
    // The fused kernel accumulates decay in a single exp pass and carries its
    // own ~1e-3 f32 error; the tensor path (burn-gdn2 0.5.1) is exact to the
    // recurrence. 2e-3 covers the kernel noise while still catching real bugs.
    assert!(
        dmax < 2e-3,
        "fused vs tensor-chunk mismatch max_diff {dmax:.3e}"
    );
}
