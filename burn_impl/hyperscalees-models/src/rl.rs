//! RL structure (`InputProcessor` / `OutputProcessor` / `ActorCriticMLP`),
//! ported from `src/hyperscalees/models/rl.py`.
//!
//! # Structural port scope
//!
//! This is a STRUCTURAL port of the layer / tensor organization and shapes,
//! not a 1:1 reproduction of the full RL stack. Two dependencies of the Python
//! source are intentionally NOT ported:
//!
//! 1. `gymnax.environments.spaces` (Discrete / Box / BoxDiscrete) is modelled
//!    by the [`Space`] enum. `Space::BoxDiscrete` (an integer `Box`) maps to
//!    [`Space::Discrete`]: only the count `n` is needed structurally — the
//!    low/high bounds only matter at the sampling layer, which is out of scope.
//! 2. `distrax` distributions are represented as raw output tensors, not
//!    probability objects:
//!    - `distrax.Categorical(logits=x)` → [`Output::Discrete`] holding the
//!      raw logits `(batch, n)`.
//!    - `distrax.MultivariateNormalDiag(mean, exp(log_std))` →
//!      [`Output::Continuous`] holding `mean` `(batch, out)` and `log_std`
//!      `(out,)` (std is `exp(log_std)`).

use burn::tensor::{Device, Tensor};
use hyperscalees_core::B;

use crate::common::{Activation, Embedding, Linear, Mm, Mlp, Parameter};

/// Apply `activation` to a 2-D tensor. Local mirror of the (private)
/// `Activation::apply` in `common`, used here because `apply` is not exposed.
fn apply_activation<const D: usize>(activation: &Activation, x: Tensor<B, D>) -> Tensor<B, D> {
    match activation {
        Activation::Relu => crate::common::relu(x),
        Activation::Silu => crate::common::silu(x),
        Activation::Pqn => crate::common::pqn(x),
        Activation::None => x,
    }
}

/// The action/observation space — a minimal stand-in for `gymnax.spaces`.
///
/// - [`Self::Discrete`]: `spaces.Discrete(n)`; also structurally covers
///   `spaces.Box` with integer dtype (`boxdiscrete`) since only the count
///   `n` matters through the network (see module docs).
/// - [`Self::Continuous`]: `spaces.Box` with float32 dtype; only the flat
///   projection size is needed (`np.prod(space.low.shape)`).
#[derive(Debug, Clone)]
pub enum Space {
    /// Discrete action/observation space of size `n`.
    Discrete { n: usize },
    /// Continuous space of flat size `size`.
    Continuous { size: usize },
}

/// The structural output of the actor head (replaces the distrax objects).
///
/// - [`Self::Discrete`]: raw logits `(batch, n)` — structural equivalent of
///   `distrax.Categorical(logits=x)`, but WITHOUT the distribution object.
/// - [`Self::Continuous`]: the mean `(batch, out)` and `log_std` `(out,)` —
///   structural equivalent of `distrax.MultivariateNormalDiag(mean,
///   exp(log_std))`, but only the raw tensors are kept (std = `exp(log_std)`).
#[derive(Debug, Clone)]
pub enum Output {
    /// Discrete-space logits `(batch, n)`.
    Discrete { logits: Tensor<B, 2> },
    /// Continuous-space mean `(batch, out)` and `log_std` `(out,)`.
    Continuous { mean: Tensor<B, 2>, log_std: Tensor<B, 1> },
}

/// What an [`InputProcessor`] actually contains.
enum InputKind {
    /// Discrete space -> embedding table lookup (es_class [`EsClass::EmbParam`]).
    Embed(Embedding),
    /// Continuous space -> matrix projection (es_class [`EsClass::MmParam`]).
    Project(Mm),
}

/// Input projection from a raw observation to the embedding dim `n_embd`.
///
/// Mirrors `InputProcessor`:
/// - discrete space -> [`Embedding`] (`Embedding(n, n_embd)`);
/// - continuous space -> [`Mm`] (`Mm(in_size, n_embd)`).
pub struct InputProcessor {
    inner: InputKind,
}

impl InputProcessor {
    /// Build an input processor for `space`, projecting to `n_embd`.
    ///
    /// Mirrors `InputProcessor.rand_init`: discrete -> `Embedding`,
    /// continuous -> `MM`.
    pub fn new(space: &Space, n_embd: usize, device: &Device<B>) -> Self {
        let inner = match space {
            Space::Discrete { n } => InputKind::Embed(Embedding::new(*n, n_embd, device)),
            Space::Continuous { size } => InputKind::Project(Mm::new(*size, n_embd, device)),
        };
        Self { inner }
    }

    /// Forward: embed (int) or project (float), yielding `(batch, n_embd)`.
    ///
    /// Mirrors `InputProcessor._forward`. The observation `x` arrives as a
    /// float tensor of shape `(batch, 1)` (discrete env); it is cast to `Int`
    /// and squeezed before the embedding row lookup.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match &self.inner {
            InputKind::Embed(emb) => {
                // Cast float observation to integer row indices, then drop the
                // trailing size-1 action dim -> (batch,).
                let idx = x.int().squeeze_dim::<1>(1);
                emb.forward(idx)
            }
            InputKind::Project(mm) => mm.forward(x),
        }
    }

    /// The ES class of the inner component.
    pub fn es_class(&self) -> crate::common::EsClass {
        match &self.inner {
            InputKind::Embed(_) => crate::common::EsClass::EmbParam,
            InputKind::Project(_) => crate::common::EsClass::MmParam,
        }
    }
}

/// Output head: discrete -> linear logits; continuous -> linear mean + log-std.
///
/// Mirrors `OutputProcessor`:
/// - discrete space -> [`Linear`]`(n_embd, n, use_bias)` (actor head);
/// - continuous space -> [`Linear`]`(n_embd, out_size, use_bias)` +
///   a [`Parameter`] `log_std = zeros(out_size)`.
pub struct OutputProcessor {
    /// Discrete-space linear head (actor logits).
    discrete: Option<Linear>,
    /// Continuous-space linear head (mean) and log-std parameter (std).
    continuous: Option<(Linear, Parameter)>,
}

impl OutputProcessor {
    /// Build an output processor for `space`, from `n_embd`.
    ///
    /// Mirrors `OutputProcessor.rand_init`.
    pub fn new(
        space: &Space,
        n_embd: usize,
        use_bias: bool,
        device: &Device<B>,
    ) -> Self {
        match space {
            Space::Discrete { n } => Self {
                discrete: Some(Linear::new(n_embd, *n, use_bias, device)),
                continuous: None,
            },
            Space::Continuous { size } => {
                let linear = Linear::new(n_embd, *size, use_bias, device);
                let log_std = Tensor::<B, 1>::zeros([*size], device);
                Self {
                    discrete: None,
                    continuous: Some((linear, Parameter::new(log_std))),
                }
            }
        }
    }

    /// Forward: logits `(batch, n)` or `(mean, log_std)`. See [`Output`].
    ///
    /// Mirrors `OutputProcessor._forward`:
    /// - discrete -> `Linear.forward(x)` → `Categorical` logits (raw return);
    /// - continuous -> `Linear.forward(x)` (mean) + `exp(log_std)` std.
    pub fn forward(&self, x: Tensor<B, 2>) -> Output {
        match &self.discrete {
            Some(linear) => {
                let logits = linear.forward(x);
                Output::Discrete { logits }
            }
            None => {
                let (linear, param) = self
                    .continuous
                    .as_ref()
                    .expect("continuous output processor must have a linear + log_std");
                let mean = linear.forward(x);
                let log_std = param.value.clone();
                Output::Continuous { mean, log_std }
            }
        }
    }

    /// True when this output processor targets a discrete space.
    pub fn is_discrete(&self) -> bool {
        self.discrete.is_some()
    }
}

/// Actor-Critic MLP network.
///
/// Mirrors `ActorCriticMLP`:
/// - `obs_embed`: [`InputProcessor`] for `obs_space -> n_embd`;
/// - `act_head`: [`OutputProcessor`] for `act_space`;
/// - `mlp`: [`Mlp`]`(n_embd, n_embd, [n_embd]*(n_layers-1), use_bias,
///   activation)`;
/// - `critic_head` (optional): [`Linear`]`(n_embd, 1, use_bias)`.
///
/// Forward: `obs_embed -> activation -> mlp -> activation -> act_head`, plus an
/// optional critic scalar `(batch, 1)`.
pub struct ActorCriticMlp {
    obs_embed: InputProcessor,
    act_head: OutputProcessor,
    mlp: Mlp,
    critic_head: Option<Linear>,
    activation: Activation,
}

impl ActorCriticMlp {
    /// Build the network. Mirrors `ActorCriticMLP.rand_init`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_embd: usize,
        obs_space: &Space,
        act_space: &Space,
        n_layers: usize,
        use_bias: bool,
        activation: Activation,
        have_critic: bool,
        device: &Device<B>,
    ) -> Self {
        let obs_embed = InputProcessor::new(obs_space, n_embd, device);
        let act_head = OutputProcessor::new(act_space, n_embd, use_bias, device);
        // `[n_embd] * (n_layers - 1)` hidden dims.
        let hidden = vec![n_embd; n_layers.saturating_sub(1)];
        let mlp = Mlp::new(n_embd, n_embd, &hidden, use_bias, activation, device);
        let critic_head = if have_critic {
            Some(Linear::new(n_embd, 1, use_bias, device))
        } else {
            None
        };
        Self {
            obs_embed,
            act_head,
            mlp,
            critic_head,
            activation,
        }
    }

    /// Forward pass: `obs_embed -> activation -> mlp -> activation ->
    /// act_head`, returning the actor [`Output`] and an optional critic scalar
    /// tensor of shape `(batch, 1)`.
    ///
    /// Mirrors `ActorCriticMLP._forward`.
    pub fn forward(&self, x: Tensor<B, 2>) -> (Output, Option<Tensor<B, 2>>) {
        let x = self.obs_embed.forward(x);
        let x = apply_activation(&self.activation, x);
        let x = self.mlp.forward(x);
        let x = apply_activation(&self.activation, x);
        let pi = self.act_head.forward(x.clone());
        let critic = self
            .critic_head
            .as_ref()
            .map(|critic_head| critic_head.forward(x));
        (pi, critic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::EsClass;

    fn device() -> Device<B> {
        Device::<B>::default()
    }

    fn to_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().into_vec::<f32>().unwrap()
    }

    fn near(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    const TOL: f32 = 1e-4;

    // -- InputProcessor: discrete -------------------------------------------

    #[test]
    fn input_processor_discrete_embeds_and_selects_rows() {
        // Hand-built embedding table (n=3, n_embd=3).
        let table = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            &device(),
        );
        let ip = InputProcessor {
            inner: InputKind::Embed(Embedding { table }),
        };
        // Observations (batch=2, 1): rows 2 and 0.
        let x = Tensor::<B, 2>::from_data([[2.0_f32], [0.0]], &device());
        let out = to_vec(ip.forward(x));
        assert_eq!(out, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]);
        assert_eq!(ip.es_class(), EsClass::EmbParam);
    }

    #[test]
    fn input_processor_discrete_new_builds_embedding() {
        let ip = InputProcessor::new(&Space::Discrete { n: 5 }, 4, &device());
        let x = Tensor::<B, 2>::from_data([[1.0_f32], [3.0]], &device());
        let out = ip.forward(x);
        assert_eq!(out.dims(), [2, 4], "batch=2, n_embd=4");
        assert_eq!(ip.es_class(), EsClass::EmbParam);
    }

    // -- InputProcessor: continuous -----------------------------------------

    #[test]
    fn input_processor_continuous_projects() {
        // Mm weight W = [[1,0],[0,1]] (out_dim=2, in_dim=2, identity).
        let w = Tensor::<B, 2>::from_data([[1.0_f32, 0.0], [0.0, 1.0]], &device());
        let ip = InputProcessor {
            inner: InputKind::Project(Mm { weight: w }),
        };
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let out = to_vec(ip.forward(x));
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ip.es_class(), EsClass::MmParam);
    }

    #[test]
    fn input_processor_continuous_new_builds_mm() {
        let ip = InputProcessor::new(&Space::Continuous { size: 3 }, 2, &device());
        let x = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &device(),
        );
        let out = ip.forward(x);
        assert_eq!(out.dims(), [2, 2], "batch=2, n_embd=2");
        assert_eq!(ip.es_class(), EsClass::MmParam);
    }

    // -- OutputProcessor: discrete ------------------------------------------

    #[test]
    fn output_processor_discrete_logits() {
        // n_embd=2 -> n=2, no bias, identity weight.
        let w = Tensor::<B, 2>::from_data([[1.0_f32, 0.0], [0.0, 1.0]], &device());
        let op = OutputProcessor {
            discrete: Some(Linear {
                weight: Mm { weight: w },
                bias: None,
            }),
            continuous: None,
        };
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let out = op.forward(x);
        match out {
            Output::Discrete { logits } => {
                let d = to_vec(logits);
                assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected discrete output, got {other:?}"),
        }
        assert!(op.is_discrete());
    }

    #[test]
    fn output_processor_discrete_new_builds_linear() {
        let op = OutputProcessor::new(&Space::Discrete { n: 3 }, 4, false, &device());
        assert!(op.is_discrete());
        let x = Tensor::<B, 2>::zeros([2, 4], &device());
        let out = op.forward(x);
        match out {
            Output::Discrete { logits } => {
                assert_eq!(logits.dims(), [2, 3], "batch=2, n actions=3");
            }
            _ => panic!("expected discrete output"),
        }
    }

    // -- OutputProcessor: continuous ----------------------------------------

    #[test]
    fn output_processor_continuous_mean_logstd() {
        // n_embd=2 -> out=2, no bias, identity weight.
        let w = Tensor::<B, 2>::from_data([[1.0_f32, 0.0], [0.0, 1.0]], &device());
        let log_std = Tensor::<B, 1>::from_data([0.0_f32, 2.0], &device());
        let op = OutputProcessor {
            discrete: None,
            continuous: Some((
                Linear {
                    weight: Mm { weight: w },
                    bias: None,
                },
                Parameter { value: log_std },
            )),
        };
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let out = op.forward(x);
        match out {
            Output::Continuous { mean, log_std } => {
                let m = to_vec(mean);
                assert_eq!(m, vec![1.0, 2.0, 3.0, 4.0]);
                let ls = to_vec(log_std);
                assert_eq!(ls, vec![0.0, 2.0]);
                // std = exp(log_std).
                let s: Vec<f32> = ls.iter().map(|v| v.exp()).collect();
                assert!(near(s[0], 1.0, TOL));
                assert!(near(s[1], 2.0_f32.exp(), TOL));
            }
            _ => panic!("expected continuous output"),
        }
        assert!(!op.is_discrete());
    }

    #[test]
    fn output_processor_continuous_new_builds_linear_and_logstd() {
        let op = OutputProcessor::new(&Space::Continuous { size: 3 }, 4, false, &device());
        assert!(!op.is_discrete());
        let x = Tensor::<B, 2>::zeros([2, 4], &device());
        let out = op.forward(x);
        match out {
            Output::Continuous { mean, log_std } => {
                assert_eq!(mean.dims(), [2, 3], "batch=2, out size=3");
                assert_eq!(log_std.dims(), [3], "log_std per action");
                // Initial log_std = zeros -> std = 1.
                let ls = to_vec(log_std);
                assert!(ls.iter().all(|v| near(*v, 0.0, TOL)));
            }
            _ => panic!("expected continuous output"),
        }
    }

    // -- ActorCriticMlp -----------------------------------------------------

    #[test]
    fn actor_critic_no_critic() {
        // obs = discrete(3), act = discrete(2), 1 layer, no critic.
        let mlp = ActorCriticMlp::new(
            4,
            &Space::Discrete { n: 3 },
            &Space::Discrete { n: 2 },
            1,
            false,
            Activation::Relu,
            false,
            &device(),
        );
        let x = Tensor::<B, 2>::from_data([[0.0_f32], [1.0]], &device());
        let (pi, critic) = mlp.forward(x);
        assert!(critic.is_none());
        match pi {
            Output::Discrete { logits } => {
                assert_eq!(logits.dims(), [2, 2], "batch=2, n actions=2");
            }
            _ => panic!("expected discrete output"),
        }
    }

    #[test]
    fn actor_critic_with_critic() {
        // obs = continuous(2), act = continuous(2), 1 layer, critic.
        let mlp = ActorCriticMlp::new(
            4,
            &Space::Continuous { size: 2 },
            &Space::Continuous { size: 2 },
            1,
            false,
            Activation::Relu,
            true,
            &device(),
        );
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let (pi, critic) = mlp.forward(x);
        let crit = critic.expect("critic present");
        assert_eq!(crit.dims(), [2, 1], "critic scalar per row (batch,1)");
        match pi {
            Output::Continuous { mean, log_std } => {
                assert_eq!(mean.dims(), [2, 2], "batch=2, out size=2");
                assert_eq!(log_std.dims(), [2], "log_std per action");
            }
            _ => panic!("expected continuous output"),
        }
    }

    #[test]
    fn actor_critic_relu_hand_computed() {
        // Fully hand-computable 1-layer net:
        //   obs_embed: embedding table [[1.0],[2.0]] (n=2 rows, n_embd=1),
        //   act_head: linear 1->1 identity, no bias,
        //   mlp: single 1->1 identity layer,
        //   activation: ReLU (applied between embed->mlp and mlp->head).
        // x=0 -> embed=1.0; relu=1.0; mlp=1.0; relu=1.0; head=1.0.
        // x=1 -> embed=2.0; relu=2.0; mlp=2.0; relu=2.0; head=2.0.
        let embed_table = Tensor::<B, 2>::from_data([[1.0_f32], [2.0]], &device());
        let obs_embed = InputProcessor {
            inner: InputKind::Embed(Embedding { table: embed_table }),
        };
        let act_w = Tensor::<B, 2>::from_data([[1.0_f32]], &device());
        let act_head = OutputProcessor {
            discrete: Some(Linear {
                weight: Mm { weight: act_w },
                bias: None,
            }),
            continuous: None,
        };
        let mlp_w = Tensor::<B, 2>::from_data([[1.0_f32]], &device());
        let mlp = Mlp {
            layers: vec![Linear {
                weight: Mm { weight: mlp_w },
                bias: None,
            }],
            activation: Activation::Relu,
        };
        let net = ActorCriticMlp {
            obs_embed,
            act_head,
            mlp,
            critic_head: None,
            activation: Activation::Relu,
        };
        let x = Tensor::<B, 2>::from_data([[0.0_f32], [1.0]], &device());
        let (pi, _critic) = net.forward(x);
        match pi {
            Output::Discrete { logits } => {
                let d = to_vec(logits);
                assert_eq!(d, vec![1.0, 2.0]);
            }
            _ => panic!("expected discrete output"),
        }
    }

    // -- ES class structural checks ------------------------------------------

    #[test]
    fn es_class_structure() {
        // The integer constants mirror the Python es_map.
        assert_eq!(EsClass::EmbParam.to_i32(), crate::common::EMB_PARAM);
        assert_eq!(EsClass::MmParam.to_i32(), crate::common::MM_PARAM);
        assert_eq!(EsClass::Param.to_i32(), crate::common::PARAM);

        // InputProcessor mirrors the right classes.
        let table = Tensor::<B, 2>::from_data([[1.0_f32]], &device());
        let ip_d = InputProcessor {
            inner: InputKind::Embed(Embedding { table }),
        };
        assert_eq!(ip_d.es_class(), EsClass::EmbParam);

        let w = Tensor::<B, 2>::from_data([[1.0_f32]], &device());
        let ip_c = InputProcessor {
            inner: InputKind::Project(Mm { weight: w }),
        };
        assert_eq!(ip_c.es_class(), EsClass::MmParam);

        // OutputProcessor linear weight is an MM_PARAM; log_std is a PARAM.
        // (`es_class` is an associated function on the component types.)
        assert_eq!(Mm::es_class(), EsClass::MmParam);
        assert_eq!(Parameter::es_class(), EsClass::Param);

        // Build an output processor over each space kind to assert its
        // structural wiring (discrete head present / continuous head + log_std
        // present) and that the underlying classes line up.
        let _op_d = OutputProcessor::new(&Space::Discrete { n: 2 }, 4, true, &device());
        assert!(OutputProcessor::new(&Space::Discrete { n: 3 }, 4, true, &device()).is_discrete());
        assert!(
            !OutputProcessor::new(&Space::Continuous { size: 3 }, 4, true, &device()).is_discrete()
        );
    }
}
