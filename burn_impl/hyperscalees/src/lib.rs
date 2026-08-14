//! HyperScaleES facade — umbrella crate re-exporting the core pieces.
//!
//! This crate is the only workspace member allowed to depend on all of
//! `hyperscalees-models`, `hyperscalees-noiser` and `hyperscalees-envs`
//! (avoiding the models <-> noiser dependency cycle), so the end-to-end
//! evolutionary training smoke driver lives here.

pub mod snn_mnist_train;

pub use hyperscalees_core::B;
