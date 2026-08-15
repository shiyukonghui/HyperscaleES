//! HyperScaleES facade — umbrella crate re-exporting the core pieces.
//!
//! This crate is the only workspace member allowed to depend on all of
//! `hyperscalees-models`, `hyperscalees-noiser` and `hyperscalees-envs`
//! (avoiding the models <-> noiser dependency cycle), so the end-to-end
//! evolutionary training smoke driver lives here.

pub mod snn_mnist_train;

/// cuBLAS 集成（einsum 同流 GEMM，仅 GPU feature 编译）。
#[cfg(feature = "gpu")]
pub mod cublas;

pub use hyperscalees_core::B;
