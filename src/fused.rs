//! Fused CUDA chunked KDA forward.
//!
//! KDA's chunkwise form (Kimi Linear Eqs 6-9) is algebraically identical to
//! the GDN-2 WY chunk algorithm under the mapping:
//!   g (log decay)  = log(alpha)        (channel-wise retention factors)
//!   b              = beta_k            (erase strength, key channels)
//!   w_gate         = beta_v            (write strength: KDA's pseudo-value is
//!                                       U = (I+T')^{-1}(β⊙V), NOT V)
//!   scale          = 1                 (no softmax scale in KDA)
//! With those inputs, `chunk_wy_forward` computes exactly
//!   A  = Tril((Q⊙Γ)(K/Γ)^T)
//!   W  = (I + T')^{-1}(β⊙Γ⊙K),  U = (I + T')^{-1}(β⊙V)
//!   S += Diag(Γ^C)S + (Γ⊙K)^T(U - W S),  O = (Γ⊙Q)S + A(U - W S)
//! so the existing GDN-2 fused kernels are reused as-is.
//!
//! The kernel is only used on the bare CUDA `CubeBackend`; everything else
//! falls back to the tensor-ops chunk path.

#[cfg(feature = "cuda")]
pub mod cuda {
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;
    use burn_cubecl::CubeBackend;
    use std::any::TypeId;

    pub type CudaBare = CubeBackend<cubecl::cuda::CudaRuntime, f32, i32, u8>;

    fn is_cuda<B: Backend>() -> bool {
        TypeId::of::<B>() == TypeId::of::<CudaBare>()
    }

    /// `q,k,v`: `[B, H, T, K/V]` projected (L2-normed q/k, Swish v).
    /// `log_alpha`: `[B, H, T, K]` log retention factors.
    /// `beta_k`: `[B, H, T, K]` write strength over key channels (erase term).
    /// `beta_v`: `[B, H, T, V]` write strength over value channels (the KDA
    /// pseudo-value is `U = (I+T')^-1 (beta ⊙ V)`, so the value gate input is
    /// `beta`, not 1).
    /// `state`: `[B, H, K, V]`.
    /// Returns `(out [B,H,T,V], state)` if the fused path applies.
    #[allow(clippy::too_many_arguments)]
    pub fn kda_fused_chunk<B: Backend>(
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        log_alpha: Tensor<B, 4>,
        beta_k: Tensor<B, 4>,
        beta_v: Tensor<B, 4>,
        state: Tensor<B, 4>,
        chunk_size: usize,
    ) -> Option<(Tensor<B, 4>, Tensor<B, 4>)> {
        if !is_cuda::<B>() {
            return None;
        }
        burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward::<B>(
            q, k, v, log_alpha, beta_k, beta_v, state, 1.0, chunk_size,
        )
    }
}
