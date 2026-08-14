//! HyperScaleES core — fundamental types shared across the workspace.

/// The backend type used throughout HyperScaleES.
///
/// In burn 0.21 the CPU backend is `burn::backend::Flex`, a flexible
/// ndarray-based backend that supports `f32`/`i32` element types.
pub type B = burn::backend::Flex;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor};

    #[test]
    fn smoke_matmul() {
        // Create two 2x2 tensors on the default device.
        let device = Device::<B>::default();
        let a = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device);
        let b = Tensor::<B, 2>::from_data([[5.0_f32, 6.0], [7.0, 8.0]], &device);

        let c = a.matmul(b);
        let expected = Tensor::<B, 2>::from_data([[19.0_f32, 22.0], [43.0, 50.0]], &device);

        assert!(c.equal(expected).all().into_scalar());
    }

    #[test]
    fn smoke_add() {
        let device = Device::<B>::default();
        let a = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 3.0], &device);
        let b = Tensor::<B, 1>::from_data([10.0_f32, 20.0, 30.0], &device);

        let c = a + b;
        let expected = Tensor::<B, 1>::from_data([11.0_f32, 22.0, 33.0], &device);

        assert!(c.equal(expected).all().into_scalar());
    }
}
