//! HyperScaleES core — fundamental types shared across the workspace.

/// 后端类型，feature 门控：
/// - 默认（`flex` feature）：CPU 后端 `burn::backend::Flex`（ndarray 系）；
/// - `gpu` feature：CUDA 后端 `burn::backend::Cuda`（CubeCL，RTX 4090 等 NVIDIA GPU）。
#[cfg(feature = "gpu")]
pub type B = burn::backend::Cuda;

/// CPU 后端（默认）。
#[cfg(not(feature = "gpu"))]
pub type B = burn::backend::Flex;

/// 返回默认计算设备：CPU 为默认 CPU 设备；GPU 为 0 号 CUDA 设备（`CudaDevice::default()`）。
///
/// 全 workspace 统一从这里取设备，避免各处散落 `Device::<B>::default()`。
pub fn default_device() -> burn::tensor::Device<B> {
    burn::tensor::Device::<B>::default()
}

/// 是否为 GPU（CUDA）后端。供训练脚本打印运行环境。
pub fn is_gpu() -> bool {
    cfg!(feature = "gpu")
}

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

        // CUDA 后端 `into_scalar` 对 bool 张量返回 u8，因此用 `!= 0` 判断真值。
        assert!(c.equal(expected).all().into_scalar() != 0);
    }

    #[test]
    fn smoke_add() {
        let device = Device::<B>::default();
        let a = Tensor::<B, 1>::from_data([1.0_f32, 2.0, 3.0], &device);
        let b = Tensor::<B, 1>::from_data([10.0_f32, 20.0, 30.0], &device);

        let c = a + b;
        let expected = Tensor::<B, 1>::from_data([11.0_f32, 22.0, 33.0], &device);

        assert!(c.equal(expected).all().into_scalar() != 0);
    }

    /// 仅在 GPU feature 开启时编译的 GPU 冒烟测试：确认 CUDA 设备可用且运算正确。
    #[cfg(feature = "gpu")]
    #[test]
    fn smoke_gpu_matmul() {
        use burn::backend::cuda::CudaDevice;
        // 显式构造 0 号 CUDA 设备，验证 GPU 路径（而非误用 CPU）。
        let device = CudaDevice::new(0);
        let a = Tensor::<B, 2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device);
        let b = Tensor::<B, 2>::from_data([[5.0_f32, 6.0], [7.0, 8.0]], &device);
        let c = a.matmul(b);
        let expected = Tensor::<B, 2>::from_data([[19.0_f32, 22.0], [43.0, 50.0]], &device);
        assert!(c.equal(expected).all().into_scalar() != 0);
        // 此测试仅在 gpu feature 下编译，确认后端确实是 CUDA。
        assert!(is_gpu());
    }
}
