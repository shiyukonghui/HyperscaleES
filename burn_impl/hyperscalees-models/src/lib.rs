//! HyperScaleES models — model layer definitions ported from
//! `src/hyperscalees/models/*.py`.
//!
//! Because burn tensors are statically ranked (unlike JAX pytrees), each model
//! is a plain Rust struct holding named `Tensor` fields. The noiser crate is a
//! sibling that perturbs these parameters at the ES level; model code injects
//! the noised matmul as a closure to avoid a dependency cycle.

pub mod common;
pub mod snn;
pub mod snn_attention;
pub mod snn_transformer;
pub mod rl;
pub mod llm;
