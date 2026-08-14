//! Common model layer definitions, ported from
//! `src/hyperscalees/models/common.py`.
//!
//! Because burn tensors are statically ranked with heterogeneous shapes
//! (unlike JAX pytrees), the model components are plain Rust structs holding
//! named `Tensor` fields. The noiser crate (a sibling crate) is responsible for
//! perturbing/clipping parameters at the ES level; here we expose the raw
//! weights so the noiser can compose noise around them.

use burn::tensor::activation;
use burn::tensor::{Device, Distribution, Int, Tensor};
use hyperscalees_core::B;

/// es_map classification: plain parameter (scalar / vector).
pub const PARAM: i32 = 0;
/// es_map classification: matrix-multiply weight.
pub const MM_PARAM: i32 = 1;
/// es_map classification: embedding table.
pub const EMB_PARAM: i32 = 2;
/// es_map classification: excluded from ES updates.
pub const EXCLUDED: i32 = 3;

/// The ES class of a component, mirroring the Python `PARAM`/`MM_PARAM`/
/// `EMB_PARAM`/`EXCLUDED` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EsClass {
    /// Plain parameter (es_map `` 0 ``).
    Param,
    /// Matrix-multiply weight (es_map `` 1 ``).
    MmParam,
    /// Embedding table (es_map `` 2 ``).
    EmbParam,
    /// Excluded from ES (es_map `` 3 ``).
    Excluded,
}

impl EsClass {
    /// The integer es_map classification.
    pub fn to_i32(self) -> i32 {
        match self {
            EsClass::Param => PARAM,
            EsClass::MmParam => MM_PARAM,
            EsClass::EmbParam => EMB_PARAM,
            EsClass::Excluded => EXCLUDED,
        }
    }
}

/// Layer normalization over the last axis:
/// ``(x - mean) / sqrt(var + eps)``.
///
/// Uses the numerically stable ``E[x^2] - (E[x])^2`` identity; ``mean_dim``
/// keeps the reduced axis as size 1 so broadcasting against ``x`` works.
pub fn layer_norm<const D: usize>(x: Tensor<B, D>, eps: f32) -> Tensor<B, D> {
    let mean = x.clone().mean_dim(-1);
    let mean_sq = mean.clone().powf_scalar(2.0);
    let x_sq = x.clone().powf_scalar(2.0);
    let var = x_sq.mean_dim(-1) - mean_sq;
    let std = (var + eps).powf_scalar(0.5);
    (x - mean) / std
}

/// Rectified linear unit.
pub fn relu<const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    activation::relu(x)
}

/// SiLU / Swish: ``x * sigmoid(x)``.
pub fn silu<const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let sig = activation::sigmoid(x.clone());
    x.mul(sig)
}

/// PQN activation: ``relu(layer_norm(x))``.
pub fn pqn<const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    relu(layer_norm(x, 1e-5))
}

/// Activation function enum used by [`Mlp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// ReLU.
    Relu,
    /// SiLU / Swish.
    Silu,
    /// PQN.
    Pqn,
    /// No activation (identity).
    None,
}

impl Activation {
    fn apply<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Activation::Relu => relu(x),
            Activation::Silu => silu(x),
            Activation::Pqn => pqn(x),
            Activation::None => x,
        }
    }
}

/// Matrix-multiply weight. Weight has shape ``(out_dim, in_dim)`` and is
/// initialized ``~ N(0, 1/sqrt(in_dim))``; es_class is [`MM_PARAM`].
pub struct Mm {
    pub weight: Tensor<B, 2>,
}

impl Mm {
    /// Create an MM weight of shape ``(out_dim, in_dim)`` scaled by
    /// ``1/sqrt(in_dim)`` (standard normal ``* scale``), mirroring
    /// `MM.rand_init`.
    pub fn new(in_dim: usize, out_dim: usize, device: &Device<B>) -> Self {
        let scale = 1.0 / (in_dim as f32).sqrt();
        let weight = Tensor::random(
            [out_dim, in_dim],
            Distribution::Normal(0.0, scale as f64),
            device,
        );
        Self { weight }
    }

    /// No-noise forward: ``x @ weight.T``, mirroring `noiser.do_mm` when no
    /// noise / iterinfo is present.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.weight.clone().transpose())
    }

    /// The ES class for MM weights.
    pub fn es_class() -> EsClass {
        EsClass::MmParam
    }
}

/// Transposed matrix-multiply weight. Weight has shape ``(in_dim, out_dim)``;
/// forward is ``x @ weight``. es_class is [`MM_PARAM`].
pub struct Tmm {
    pub weight: Tensor<B, 2>,
}

impl Tmm {
    /// Create a TMM weight of shape ``(in_dim, out_dim)`` scaled by
    /// ``1/sqrt(in_dim)``.
    pub fn new(in_dim: usize, out_dim: usize, device: &Device<B>) -> Self {
        let scale = 1.0 / (in_dim as f32).sqrt();
        let weight = Tensor::random(
            [in_dim, out_dim],
            Distribution::Normal(0.0, scale as f64),
            device,
        );
        Self { weight }
    }

    /// No-noise forward: ``x @ weight``.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.weight.clone())
    }

    /// The ES class for TMM weights.
    pub fn es_class() -> EsClass {
        EsClass::MmParam
    }
}

/// A linear layer: an [`Mm`] weight with an optional bias (a rank-1
/// [`PARAM`]-class vector), mirroring `Linear` in Python.
pub struct Linear {
    pub weight: Mm,
    pub bias: Option<Tensor<B, 1>>,
}

impl Linear {
    /// Build a linear layer. When ``use_bias`` the bias is initialized to
    /// zeros (matching Python).
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        use_bias: bool,
        device: &Device<B>,
    ) -> Self {
        let weight = Mm::new(in_dim, out_dim, device);
        let bias = if use_bias {
            Some(Tensor::<B, 1>::zeros([out_dim], device))
        } else {
            None
        };
        Self { weight, bias }
    }

    /// Forward: ``weight.forward(x) + bias`` (bias broadcast over batch).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = self.weight.forward(x);
        match &self.bias {
            Some(bias) => {
                let bias = bias.clone().unsqueeze::<2>();
                out + bias
            }
            None => out,
        }
    }
}

/// A multi-layer perceptron: a list of [`Linear`] layers with the configured
/// activation applied between layers (the last layer has no activation),
/// mirroring `MLP` in Python.
pub struct Mlp {
    pub layers: Vec<Linear>,
    pub activation: Activation,
}

impl Mlp {
    /// Build an MLP with input dim ``in_dim``, output dim ``out_dim`` and the
    /// given hidden dims.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        hidden_dims: &[usize],
        use_bias: bool,
        activation: Activation,
        device: &Device<B>,
    ) -> Self {
        let mut input_dims = Vec::with_capacity(hidden_dims.len() + 1);
        input_dims.push(in_dim);
        input_dims.extend_from_slice(hidden_dims);

        let mut output_dims = Vec::with_capacity(hidden_dims.len() + 1);
        output_dims.extend_from_slice(hidden_dims);
        output_dims.push(out_dim);

        let layers = input_dims
            .iter()
            .zip(output_dims.iter())
            .map(|(&i, &o)| Linear::new(i, o, use_bias, device))
            .collect();

        Self { layers, activation }
    }

    /// Forward: apply each layer, applying the activation between all but the
    /// last layer (the last layer has no activation).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = self.layers.len();
        let mut out = x;
        for (i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(out);
            if i != n - 1 {
                out = self.activation.apply(out);
            }
        }
        out
    }
}

/// A plain parameter (scalar / vector), mirroring `Parameter` in Python. Its
/// es_class is [`PARAM`]. Represented as a rank-1 tensor.
pub struct Parameter {
    pub value: Tensor<B, 1>,
}

impl Parameter {
    /// Build a parameter from an explicit value.
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }

    /// The ES class for plain parameters.
    pub fn es_class() -> EsClass {
        EsClass::Param
    }
}

/// An embedding lookup table with shape ``(in_dim, out_dim)``, mirroring
/// `Embedding` in Python. Its es_class is [`EMB_PARAM`].
pub struct Embedding {
    pub table: Tensor<B, 2>,
}

impl Embedding {
    /// Build an embedding table ``(in_dim, out_dim)`` scaled by ``1/sqrt(in_dim)``.
    pub fn new(in_dim: usize, out_dim: usize, device: &Device<B>) -> Self {
        let scale = 1.0 / (in_dim as f32).sqrt();
        let table = Tensor::random(
            [in_dim, out_dim],
            Distribution::Normal(0.0, scale as f64),
            device,
        );
        Self { table }
    }

    /// Forward: gather table rows by index, mirroring `do_emb` / ``param[x]``.
    pub fn forward(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        self.table.clone().select(0, indices)
    }

    /// The ES class for embedding tables.
    pub fn es_class() -> EsClass {
        EsClass::EmbParam
    }
}

// ---------------------------------------------------------------------------
// Structural analogues of the JAX key-tree helpers.
//
// JAX's `simple_es_tree_key` produces a parallel pytree of PRNG keys (one per
// parameter leaf) and `recursive_scan_split` splits each key along scan axes.
// Because burn does not have JAX-compatible PRNGs (and the structs here are
// not pytrees) we provide structural equivalents: they reproduce the *shape /
// count* semantics without attempting to match JAX key values.
// ---------------------------------------------------------------------------

/// Produce one key per component leaf. Structural analogue of
/// `simple_es_tree_key`: returns exactly `num_leaves` keys derived from
/// `base_key`. Keys are deterministic and distinct for the given count.
pub fn simple_es_tree_key(num_leaves: usize, base_key: u64) -> Vec<u64> {
    (0..num_leaves)
        .map(|i| base_key.wrapping_add(i as u64 + 1))
        .collect()
}

/// Total number of keys produced by recursively splitting a parameter of
/// `shape` over the given (pre-sorted) scan axes. Structural analogue of
/// `recursive_scan_split`: an empty axis list means no split (1 key),
/// otherwise each scan axis multiplies the key count by that axis' size.
pub fn recursive_scan_split_count(shape: &[usize], scan_axes: &[usize]) -> usize {
    let mut count = 1;
    for &axis in scan_axes {
        count *= shape.get(axis).copied().unwrap_or(1);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

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

    // -- layer_norm ---------------------------------------------------------

    #[test]
    fn layer_norm_zero_mean_unit_std() {
        let x = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
            &device(),
        );
        let out = layer_norm(x, 1e-5);
        let data = to_vec(out);

        // Row 0: mean 2.5, std ~1.118; Row 1: mean 25, std ~11.18.
        let expected = [
            -1.3416406, -0.4472136, 0.4472136, 1.3416406, // row 0
            -1.3416406, -0.4472136, 0.4472136, 1.3416406, // row 1 (same normalized)
        ];
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!(near(*a, *b, TOL), "expected {b}, got {a}");
        }
    }

    // -- activations --------------------------------------------------------

    #[test]
    fn relu_clamps_negatives() {
        let x = Tensor::<B, 1>::from_data([-2.0_f32, -0.5, 0.0, 3.0], &device());
        let out = to_vec(relu(x));
        assert_eq!(out, vec![0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn silu_matches_x_sigmoid() {
        let x = Tensor::<B, 1>::from_data([-2.0_f32, 0.0, 2.0], &device());
        let out = to_vec(silu(x));
        // sigmoid(-2)=0.1192 -> -0.2384; sigmoid(0)=0.5 -> 0; sigmoid(2)=0.8808 -> 1.7616
        assert!(near(out[0], -0.23840584, TOL));
        assert!(near(out[1], 0.0, TOL));
        assert!(near(out[2], 1.7615943, TOL));
    }

    #[test]
    fn pqn_applies_relu_after_layer_norm() {
        let x = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 3.0], &device());
        let out = to_vec(pqn(x));
        // mean=2, var=E[x²]-mean² = 14/3-4 = 2/3, std=sqrt(2/3+eps)~0.8165.
        // normalized -> [-1.2247, 0, 1.2247], relu -> [0, 0, 1.2247].
        assert!(near(out[0], 0.0, TOL));
        assert!(near(out[1], 0.0, TOL));
        assert!(near(out[2], 1.2247, TOL));
    }

    // -- MM -----------------------------------------------------------------

    #[test]
    fn mm_forward_matches_expected() {
        // weight W = [[1, 0], [0, 1]] -> out_dim=2, in_dim=2, identity.
        let w = Tensor::<B, 2>::from_data([[1.0_f32, 0.0], [0.0, 1.0]], &device());
        let mm = Mm { weight: w };
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let out = to_vec(mm.forward(x));
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);

        // Non-square: W = [[1,2,3],[4,5,6]] in_dim=3 out_dim=2.
        let w = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &device(),
        );
        let mm = Mm { weight: w };
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 1.0, 1.0], [2.0, 2.0, 2.0]], &device());
        // x @ W.T = [[6, 15], [12, 30]]
        let out = to_vec(mm.forward(x));
        assert_eq!(out, vec![6.0, 15.0, 12.0, 30.0]);
    }

    #[test]
    fn mm_rand_init_scale_variance() {
        // Statistical check: with scale = 1/sqrt(in_dim), sample variance of
        // the weights should be ~ scale^2. Use a large-ish sample.
        let in_dim = 16usize;
        let out_dim = 256usize;
        let mm = Mm::new(in_dim, out_dim, &device());
        let n = (in_dim * out_dim) as f32;
        let data = to_vec(mm.weight.clone());
        let mean: f32 = data.iter().sum::<f32>() / n;
        let var: f32 = data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let scale = 1.0 / (in_dim as f32).sqrt();
        let expected_var = scale * scale;
        // Tolerance generous for a statistical test.
        assert!(
            (var - expected_var).abs() < 0.2 * expected_var,
            "var {var} vs expected {expected_var}"
        );
    }

    // -- Linear -------------------------------------------------------------

    #[test]
    fn linear_without_bias() {
        let w = Tensor::<B, 2>::from_data([[1.0_f32, 0.0], [0.0, 1.0]], &device());
        let layer = Linear {
            weight: Mm { weight: w },
            bias: None,
        };
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let out = to_vec(layer.forward(x));
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn linear_with_bias() {
        let w = Tensor::<B, 2>::from_data([[1.0_f32, 0.0], [0.0, 1.0]], &device());
        let bias = Tensor::<B, 1>::from_data([10.0_f32, 20.0], &device());
        let layer = Linear {
            weight: Mm { weight: w },
            bias: Some(bias),
        };
        let x = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device());
        let out = to_vec(layer.forward(x));
        assert_eq!(out, vec![11.0, 22.0, 13.0, 24.0]);
    }

    // -- MLP ----------------------------------------------------------------

    #[test]
    fn mlp_relu_applies_only_between_layers() {
        // 1x1 MLP with two layers, both identity weights and no bias.
        // in_dim=1, hidden=[1], out_dim=1, 2 layers.
        let w1 = Tensor::<B, 2>::from_data([[1.0_f32]], &device());
        let w2 = Tensor::<B, 2>::from_data([[1.0_f32]], &device());
        let mlp = Mlp {
            layers: vec![
                Linear {
                    weight: Mm { weight: w1 },
                    bias: None,
                },
                Linear {
                    weight: Mm { weight: w2 },
                    bias: None,
                },
            ],
            activation: Activation::Relu,
        };

        // x = -3: layer0 out = -3; first-layer applies relu -> 0; layer1 out = 0.
        let x = Tensor::<B, 2>::from_data([[-3.0_f32]], &device());
        let out = to_vec(mlp.forward(x));
        assert_eq!(out, vec![0.0]);

        // x = 5: layer0 = 5; relu -> 5; layer1 = 5.
        let x = Tensor::<B, 2>::from_data([[5.0_f32]], &device());
        let out = to_vec(mlp.forward(x));
        assert_eq!(out, vec![5.0]);
    }

    #[test]
    fn mlp_last_layer_not_activated() {
        // Verify the last layer does NOT apply relu. Use a linear layer with
        // a negative slope so relu would clamp it.
        // Layer0: identity (out = x). Layer1: weight = [[1]], so out = x.
        // With x=5 and relu between, then layer1 out = 5 (no relu after).
        let w1 = Tensor::<B, 2>::from_data([[1.0_f32]], &device());
        let w2 = Tensor::<B, 2>::from_data([[1.0_f32]], &device());
        let mlp = Mlp {
            layers: vec![
                Linear {
                    weight: Mm { weight: w1 },
                    bias: None,
                },
                Linear {
                    weight: Mm { weight: w2 },
                    bias: None,
                },
            ],
            activation: Activation::Relu,
        };
        // If relu WERE applied to the last layer with a negative-weight layer,
        // the result would differ. Here: layer0(5)=5, relu->5, layer1=5.
        let x = Tensor::<B, 2>::from_data([[5.0_f32]], &device());
        let out = to_vec(mlp.forward(x));
        assert_eq!(out, vec![5.0]);

        // Negative case proving activation is applied before the last layer:
        // x = -5 -> layer0=-5, relu->0, layer1=0. Without the inter-layer
        // activation this would be -5.
        let x = Tensor::<B, 2>::from_data([[-5.0_f32]], &device());
        let out = to_vec(mlp.forward(x));
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn mlp_multi_dim_hand_computed() {
        // Layer0 weight identity 3->3 with relu; Layer1 weight [[1,0,0]] 3->1.
        // x = [2, -3, 4] -> layer0 = [2,-3,4]; relu -> [2,0,4];
        // layer1 = 1*2 + 0*0 + 0*4 = 2. No relu on the last layer.
        let w0 = Tensor::<B, 2>::from_data(
            [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &device(),
        );
        let w1 = Tensor::<B, 2>::from_data([[1.0_f32, 0.0, 0.0]], &device());
        let mlp = Mlp {
            layers: vec![
                Linear {
                    weight: Mm { weight: w0 },
                    bias: None,
                },
                Linear {
                    weight: Mm { weight: w1 },
                    bias: None,
                },
            ],
            activation: Activation::Relu,
        };
        let x = Tensor::<B, 2>::from_data([[2.0_f32, -3.0, 4.0]], &device());
        let out = to_vec(mlp.forward(x));
        assert_eq!(out, vec![2.0]);
    }

    // -- es_map classifications --------------------------------------------

    #[test]
    fn es_classes_map_to_constants() {
        assert_eq!(Mm::es_class().to_i32(), MM_PARAM);
        assert_eq!(Tmm::es_class().to_i32(), MM_PARAM);
        assert_eq!(Embedding::es_class().to_i32(), EMB_PARAM);
        assert_eq!(Parameter::es_class().to_i32(), PARAM);
    }

    // -- Embedding ----------------------------------------------------------

    #[test]
    fn embedding_selects_rows() {
        let table = Tensor::<B, 2>::from_data(
            [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            &device(),
        );
        let emb = Embedding { table };
        // Select rows 2 and 0.
        let indices = Tensor::<B, 1, Int>::from_data([2, 0], &device());
        let out = to_vec(emb.forward(indices));
        assert_eq!(out, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]);
    }

    // -- structural key-tree analogues --------------------------------------

    #[test]
    fn simple_es_tree_key_count_matches_leaves() {
        // 5 leaves -> 5 keys, all distinct.
        let keys = simple_es_tree_key(5, 42);
        assert_eq!(keys.len(), 5);
        let mut sorted = keys.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 5);
    }

    #[test]
    fn recursive_scan_split_count_shape_parity() {
        // No split -> 1.
        assert_eq!(recursive_scan_split_count(&[4, 16], &[]), 1);
        // Split over axis 0 (size 4) -> 4.
        assert_eq!(recursive_scan_split_count(&[4, 16], &[0]), 4);
        // Split over axis 0 and 1 -> 4 * 16 = 64.
        assert_eq!(recursive_scan_split_count(&[4, 16], &[0, 1]), 64);
    }
}
