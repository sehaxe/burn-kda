#!/usr/bin/env python3
"""KDA benchmark: burn-kda (fused op / kernels) vs fla chunk_kda (Triton).

Same math (K3): q,k L2-normed, channel-wise decay with A_log/dt_bias and the
lower-bound -5 gate, beta sigmoid. Same configs, same GPU.
Run burn side: cargo test --release --features "cuda,autodiff" -p burn-kda --test bench_cuda -- --ignored --nocapture
"""
import time
import torch
from fla.ops.kda import chunk_kda


def bench(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    torch.manual_seed(0)
    for d, h, hk, T, iters in [
        (512, 4, 64, 1024, 10),
        (1024, 8, 64, 2048, 5),
        (2048, 8, 128, 4096, 3),
        (4096, 8, 128, 8192, 2),
    ]:
        kd, vd = h * hk, hk
        q = torch.randn(1, T, h, hk, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, T, h, hk, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, T, h, vd, dtype=torch.bfloat16, device="cuda")
        g = torch.randn(1, T, h, hk, dtype=torch.bfloat16, device="cuda")
        beta = torch.rand(1, T, h, dtype=torch.bfloat16, device="cuda")
        A_log = torch.randn(h, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(h * hk, dtype=torch.float32, device="cuda")
        kwargs = dict(
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=32,
        )

        def fwd():
            return chunk_kda(q, k, v, g, beta, **kwargs)

        def train():
            qg = q.detach().requires_grad_(True)
            kg = k.detach().requires_grad_(True)
            vg = v.detach().requires_grad_(True)
            gg = g.detach().requires_grad_(True)
            bg = beta.detach().requires_grad_(True)
            o, _ = chunk_kda(qg, kg, vg, gg, bg, **kwargs)
            o.pow(2).mean().backward()

        torch.cuda.reset_peak_memory_stats()
        t = bench(fwd, iters)
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"d={d} h={h} hk={hk} T={T}: fwd {t*1e3:8.1f} ms {T/t:>10,.0f} tok/s  peak={peak:.0f} MB")
        t = bench(train, max(1, iters // 2))
        print(f"{'':14}      train {t*1e3:8.1f} ms {T/t:>10,.0f} tok/s")


if __name__ == "__main__":
    main()
