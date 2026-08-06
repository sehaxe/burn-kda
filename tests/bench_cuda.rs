#![cfg(all(feature = "cuda", feature = "autodiff"))]
//! Attention-core benchmark: burn-kda fused kernels + fused autodiff op vs
//! fla chunk_kda (see bench_kda.py). Same scope: the chunked delta-rule
//! attention only (no projections / conv / gate).

use burn::tensor::{Device, Distribution, Tensor};
use burn_kda::fused::cuda::{kda_fused_chunk, CudaBare};

fn time_it(runs: usize, mut f: impl FnMut()) -> f64 {
    let t0 = std::time::Instant::now();
    for _ in 0..runs {
        f();
    }
    t0.elapsed().as_secs_f64() / runs as f64
}

#[test]
#[ignore]
fn bench_cuda() {
    let plain: Device = Default::default();
    let ad_dev = Device::autodiff(plain.clone());
    for (d, h, hk, t) in [
        (512usize, 4usize, 64usize, 1024usize),
        (1024, 8, 64, 2048),
        (2048, 8, 128, 4096),
        (4096, 8, 128, 8192),
    ] {
        let mk =
            |shape: [usize; 4]| Tensor::<4>::random(shape, Distribution::Normal(0.0, 0.1), &plain);
        let q = mk([1, h, t, hk]);
        let k = mk([1, h, t, hk]);
        let v = mk([1, h, t, hk]);
        let g = mk([1, h, t, hk]);
        let b = mk([1, h, t, hk]);
        let w = mk([1, h, t, hk]);
        let s = mk([1, h, hk, hk]);
        let mkad =
            |t: Tensor<4>| Tensor::<4>::from_data(t.clone().into_data(), &ad_dev).require_grad();
        let qa = mkad(q.clone());
        let ka = mkad(k.clone());
        let va = mkad(v.clone());
        let ga = mkad(g.clone());
        let ba = mkad(b.clone());
        let wa = mkad(w.clone());
        let sa = mkad(s.clone());

        // double warmup: the first launch per comptime config pays JIT
        for _ in 0..3 {
            let _ = kda_fused_chunk::<CudaBare>(
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                b.clone(),
                w.clone(),
                s.clone(),
                16,
            )
            .unwrap()
            .0
            .into_data();
        }
        let _ = burn_gdn2::chunk_wy_forward_autodiff::<CudaBare>(
            qa.clone(),
            ka.clone(),
            va.clone(),
            ga.clone(),
            ba.clone(),
            wa.clone(),
            sa.clone(),
            1.0,
            16,
        );
        for _ in 0..3 {
            let loss = burn_gdn2::chunk_wy_forward_autodiff::<CudaBare>(
                qa.clone(),
                ka.clone(),
                va.clone(),
                ga.clone(),
                ba.clone(),
                wa.clone(),
                sa.clone(),
                1.0,
                16,
            )
            .unwrap()
            .0
            .powf_scalar(2.0)
            .mean();
            let _ = loss.clone().into_data();
            let _g = loss.backward();
        }
        let runs = if t <= 2048 { 10 } else { 5 };
        let t_fwd = time_it(runs, || {
            let _ = kda_fused_chunk::<CudaBare>(
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                b.clone(),
                w.clone(),
                s.clone(),
                16,
            )
            .unwrap()
            .0
            .into_data();
        });
        println!(
            "d={d} h={h} hk={hk} T={t}: fused fwd {:.2} ms {:>12.0} tok/s",
            t_fwd * 1e3,
            t as f64 / t_fwd
        );

        let t_tr = time_it(runs, || {
            let loss = burn_gdn2::chunk_wy_forward_autodiff::<CudaBare>(
                qa.clone(),
                ka.clone(),
                va.clone(),
                ga.clone(),
                ba.clone(),
                wa.clone(),
                sa.clone(),
                1.0,
                16,
            )
            .unwrap()
            .0
            .powf_scalar(2.0)
            .mean();
            let _ = loss.clone().into_data();
            let _g = loss.backward();
        });
        println!(
            "{:14}      train {:.2} ms {:>12.0} tok/s",
            "",
            t_tr * 1e3,
            t as f64 / t_tr
        );
    }
}
