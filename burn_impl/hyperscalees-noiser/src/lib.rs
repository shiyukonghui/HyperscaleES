//! HyperScaleES noise generation strategies, ported from
//! `src/hyperscalees/noiser/`.

pub mod alteggroll;
pub mod eggroll;
pub mod eggroll_bs;
pub mod noiser;
pub mod open_es;
pub mod sparse;

pub use alteggroll::AltEggRoll;
pub use eggroll::{init_noiser, EggRoll};
pub use eggroll_bs::EggRollBS;
pub use noiser::{
    convert_fitnesses_impl, noise_seed, BaseNoiser, DeterministicNoise, FrozenNoiserParams,
    IterInfo, Noiser, NoiserParams, OptimizerState, Solver,
};
pub use open_es::OpenES;
pub use sparse::Sparse;
