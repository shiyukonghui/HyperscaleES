//! 临时探针：隔离 cuLaunchKernel INVALID_VALUE 问题。
//! 分别测试：1) vecadd.ptx（已知可用）经我们的 launch 封装；2) prng PTX。

use burn::backend::Cuda;
use burn::tensor::{Device, Distribution, Tensor};
use hyperscalees::oxide::{self, OxideKernel};

fn main() {
    let device = Device::<Cuda>::default();

    // ============ 0) 最先调用 oxide::prng_normal_half（state 首次初始化，任何 launch 之前）============
    let b_first: Tensor<Cuda, 3> = Tensor::empty([8, 64, 64], &device);
    match hyperscalees::oxide::prng_normal_half(&b_first, 0.0, 1.0, &device) {
        Ok(()) => {
            let oxv = b_first.clone().into_data().into_vec::<f32>().unwrap();
            let m = oxv.iter().sum::<f32>() / oxv.len() as f32;
            let v = oxv.iter().map(|x| x * x).sum::<f32>() / oxv.len() as f32 - m * m;
            println!("[probe] 最先调用 prng_normal_half OK mean={m:.4} var={v:.4}");
        }
        Err(e) => println!("[probe] 最先调用 prng_normal_half 失败: {e}"),
    }
    // 同场景手动对照：oxide::load_kernel + 探针自建流 + set_current
    let st0 = hyperscalees::cublas::state_pub(&device);
    unsafe {
        let mut cur: *mut cudarc::driver::sys::CUctx_st = std::ptr::null_mut();
        cudarc::driver::sys::cuCtxGetCurrent(&mut cur);
        println!("[probe] state 的 oxide_stream = {:?}, 当前 ctx = {cur:?}, st.ctx = {:?}", st0.oxide_stream, st0.ctx);
    }
    let ptx0 = std::fs::read("F:/PythonProject/HyperscaleES/burn_impl/hyperscalees-kernels/ptx/prng_normal_half.ptx")
        .expect("读 prng.ptx 失败");
    let mut ptx0 = ptx0.clone();
    ptx0.push(0);
    let k0 = oxide::load_kernel(&device, &ptx0, "prng_normal_half").expect("load");
    unsafe {
        let mut cur: *mut cudarc::driver::sys::CUctx_st = std::ptr::null_mut();
        cudarc::driver::sys::cuCtxGetCurrent(&mut cur);
        println!("[probe] 创建 s_new 前 current ctx = {cur:?}");
    }
    let s_new = cudarc::driver::result::stream::create(
        cudarc::driver::result::stream::StreamKind::NonBlocking,
    )
    .expect("new stream");
    println!("[probe] s_new = {s_new:?}");
    let co0 = hyperscalees::cublas::as_cube(&b_first);
    let mut p0 = hyperscalees::cublas::raw_ptr(&co0, &device) as *mut f32;
    let mut th0 = 256u32;
    let mut me0 = 0.0f32;
    let mut sd0 = 1.0f32;
    let mut a0 = 1u32;
    let mut a1 = 2u32;
    let mut a2 = 3u32;
    let mut a3 = 4u32;
    let mut pa0: [*mut std::ffi::c_void; 8] = [
        &mut p0 as *mut _ as *mut std::ffi::c_void,
        &mut th0 as *mut u32 as *mut std::ffi::c_void,
        &mut me0 as *mut f32 as *mut std::ffi::c_void,
        &mut sd0 as *mut f32 as *mut std::ffi::c_void,
        &mut a0 as *mut u32 as *mut std::ffi::c_void,
        &mut a1 as *mut u32 as *mut std::ffi::c_void,
        &mut a2 as *mut u32 as *mut std::ffi::c_void,
        &mut a3 as *mut u32 as *mut std::ffi::c_void,
    ];
    unsafe {
        cudarc::driver::result::ctx::set_current(st0.ctx).expect("ctx");
        let status = cudarc::driver::sys::cuLaunchKernel(
            oxide::kernel_function(&k0),
            1, 1, 1, 256, 1, 1, 0,
            s_new,
            pa0.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        println!("[probe] 同场景手动 launch(新流): {status:?}");
        let _ = cudarc::driver::sys::cuCtxSynchronize();
        let d0 = b_first.clone().into_data().into_vec::<f32>().unwrap();
        let m = d0.iter().sum::<f32>() / d0.len() as f32;
        let v = d0.iter().map(|x| x * x).sum::<f32>() / d0.len() as f32 - m * m;
        println!("[probe]   手动 mean={m:.4} var={v:.4}");
    }
    // 用 oxide::launch_on_stream 传探针自建流 s_new（隔离 stream 来源）
    match unsafe {
        oxide::launch_on_stream(
            &k0,
            &device,
            s_new,
            1, 1, 1, 256, 1, 1, 0,
            &mut pa0,
        )
    } {
        Ok(()) => println!("[probe] oxide::launch_on_stream(s_new) OK"),
        Err(e) => println!("[probe] oxide::launch_on_stream(s_new) 失败: {e}"),
    }
    return;

    // 1) vecadd.ptx（cuda-oxide 官方示例产物，无 libdevice）。
    let vecadd_ptx = std::fs::read("F:/PythonProject/HyperscaleES/cuda-oxide-0.2.1/cuda-oxide-0.2.1/crates/rustc-codegen-cuda/examples/vecadd/vecadd.ptx")
        .expect("读 vecadd.ptx 失败");
    println!("[probe] vecadd.ptx: {} bytes", vecadd_ptx.len());
    let kernel: OxideKernel = match oxide::load_kernel(&device, &vecadd_ptx, "vecadd") {
        Ok(k) => {
            println!("[probe] vecadd 加载成功");
            k
        }
        Err(e) => {
            println!("[probe] vecadd 加载失败: {e}");
            return;
        }
    };

    // vecadd(a, b, c)：3 个 slice 参数 = PTX 里 (ptr, len) 各一对，共 6 个 .param。
    let a: Tensor<Cuda, 2> = Tensor::random([4, 256], Distribution::Normal(0.0, 1.0), &device);
    let b: Tensor<Cuda, 2> = Tensor::random([4, 256], Distribution::Normal(0.0, 1.0), &device);
    let mut c: Tensor<Cuda, 2> = Tensor::zeros([4, 256], &device);
    let ca = hyperscalees::cublas::as_cube(&a);
    let cb = hyperscalees::cublas::as_cube(&b);
    let cc = hyperscalees::cublas::as_cube(&c);
    let mut pa = hyperscalees::cublas::raw_ptr(&ca, &device) as *mut std::ffi::c_void;
    let mut pb = hyperscalees::cublas::raw_ptr(&cb, &device) as *mut std::ffi::c_void;
    let mut pc = hyperscalees::cublas::raw_ptr(&cc, &device) as *mut std::ffi::c_void;
    let mut la = 1024u64; // 4*256
    let mut lb = 1024u64;
    let mut lc = 1024u64;
    let mut args: [*mut std::ffi::c_void; 6] = [
        &mut pa as *mut _ as *mut std::ffi::c_void,
        &mut la as *mut u64 as *mut std::ffi::c_void,
        &mut pb as *mut _ as *mut std::ffi::c_void,
        &mut lb as *mut u64 as *mut std::ffi::c_void,
        &mut pc as *mut _ as *mut std::ffi::c_void,
        &mut lc as *mut u64 as *mut std::ffi::c_void,
    ];
    // 三种 stream 对比：null / cubecl raw_stream / cudarc 自建流。
    let st = hyperscalees::cublas::state_pub(&device);
    let dh_raw = cubecl::device_handle::DeviceHandle::<<cubecl::cuda::CudaRuntime as cubecl::Runtime>::Server>::new(
        cubecl::device::Device::to_id(&device),
    );
    let cube_stream = unsafe {
        dh_raw.submit_blocking(|s| s.raw_stream(cubecl::stream_id::StreamId::current()) as usize)
            .expect("stream") as *mut cudarc::driver::sys::CUstream_st
    };
    println!("[probe] cube_stream = {cube_stream:?} (早期取)");
    let cudarc_stream = cudarc::driver::result::stream::create(
        cudarc::driver::result::stream::StreamKind::NonBlocking,
    )
    .expect("cudarc stream");
    for (name, s) in [
        ("null", std::ptr::null_mut()),
        ("cubecl-raw", cube_stream),
        ("cudarc-new", cudarc_stream),
    ] {
        unsafe {
            cudarc::driver::result::ctx::set_current(st.ctx).expect("ctx");
            let status = cudarc::driver::sys::cuLaunchKernel(
                oxide::kernel_function(&kernel),
                1, 1, 1, 256, 1, 1, 0,
                s,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            println!("[probe] launch on {name}: {status:?}");
            if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                continue;
            }
            // 同步后校验
            let _ = cudarc::driver::sys::cuCtxSynchronize();
            let sum: f32 = c.clone().sum().into_scalar();
            println!("[probe]   sum = {sum:.3}");
        }
    }
    // ============ 模拟 noise_bench 前置状态：先跑一次 cuBLAS gemm ============
    let am: Tensor<Cuda, 2> = Tensor::random([64, 784], Distribution::Normal(0.0, 1.0), &device);
    let bm: Tensor<Cuda, 2> = Tensor::random([784, 128], Distribution::Normal(0.0, 1.0), &device);
    let _cm = hyperscalees::cublas::gemm(&am, &bm, &device);
    let _s: f32 = _cm.clone().sum().into_scalar();
    println!("[probe] 前置 cuBLAS gemm 完成");

    // ============ prng_normal_half PTX 对比 ============
    let prng_ptx = std::fs::read("F:/PythonProject/HyperscaleES/burn_impl/hyperscalees-kernels/ptx/prng_normal_half.ptx")
        .expect("读 prng.ptx 失败");
    let mut prng_ptx = prng_ptx.clone();
    prng_ptx.push(0); // NUL 结尾（cuModuleLoadData 要求）
    let pk = match oxide::load_kernel(&device, &prng_ptx, "prng_normal_half") {
        Ok(k) => {
            println!("[probe] prng 加载成功 ({} bytes)", prng_ptx.len());
            k
        }
        Err(e) => {
            println!("[probe] prng 加载失败: {e}");
            return;
        }
    };
    // 对比：empty（可能未分配）vs zeros（已分配）。
    for (name, out) in [
        ("empty", Tensor::<Cuda, 3>::empty([16, 64, 64], &device)),
        ("zeros", Tensor::<Cuda, 3>::zeros([16, 64, 64], &device)),
    ] {
        let co = hyperscalees::cublas::as_cube(&out);
        let mut po = hyperscalees::cublas::raw_ptr(&co, &device) as *mut f32;
        println!("[probe] {name} raw_ptr = {po:?}");
        let mut threads = 512u32;
        let mut mean = 0.0f32;
        let mut std = 1.0f32;
        let mut s0 = 11u32;
        let mut s1 = 22u32;
        let mut s2 = 33u32;
        let mut s3 = 44u32;
        let mut pargs: [*mut std::ffi::c_void; 8] = [
            &mut po as *mut _ as *mut std::ffi::c_void,
            &mut threads as *mut u32 as *mut std::ffi::c_void,
            &mut mean as *mut f32 as *mut std::ffi::c_void,
            &mut std as *mut f32 as *mut std::ffi::c_void,
            &mut s0 as *mut u32 as *mut std::ffi::c_void,
            &mut s1 as *mut u32 as *mut std::ffi::c_void,
            &mut s2 as *mut u32 as *mut std::ffi::c_void,
            &mut s3 as *mut u32 as *mut std::ffi::c_void,
        ];
        unsafe {
            cudarc::driver::result::ctx::set_current(st.ctx).expect("ctx");
            let status = cudarc::driver::sys::cuLaunchKernel(
                oxide::kernel_function(&pk),
                2, 1, 1, 256, 1, 1, 0,
                cudarc_stream,
                pargs.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            println!("[probe] prng launch on {name}: {status:?}");
        }
    }

    // ============ 直接调用 oxide::prng_normal_half 包装 ============
    let b_ox: Tensor<Cuda, 3> = Tensor::empty([8, 64, 64], &device);
    match hyperscalees::oxide::prng_normal_half(&b_ox, 0.0, 1.0, &device) {
        Ok(()) => {
            let oxv = b_ox.clone().into_data().into_vec::<f32>().unwrap();
            let m = oxv.iter().sum::<f32>() / oxv.len() as f32;
            let v = oxv.iter().map(|x| x * x).sum::<f32>() / oxv.len() as f32 - m * m;
            println!("[probe] oxide::prng_normal_half OK mean={m:.4} var={v:.4}");
        }
        Err(e) => println!("[probe] oxide::prng_normal_half 失败: {e}"),
    }
    // prng × cubecl-raw 流组合（隔离 stream 假设）
    let co2 = hyperscalees::cublas::as_cube(&b_ox);
    let mut po2 = hyperscalees::cublas::raw_ptr(&co2, &device) as *mut f32;
    let mut threads2 = 256u32;
    let mut mean2 = 0.0f32;
    let mut std2 = 1.0f32;
    let mut t0 = 1u32;
    let mut t1 = 2u32;
    let mut t2 = 3u32;
    let mut t3 = 4u32;
    let mut pargs2: [*mut std::ffi::c_void; 8] = [
        &mut po2 as *mut _ as *mut std::ffi::c_void,
        &mut threads2 as *mut u32 as *mut std::ffi::c_void,
        &mut mean2 as *mut f32 as *mut std::ffi::c_void,
        &mut std2 as *mut f32 as *mut std::ffi::c_void,
        &mut t0 as *mut u32 as *mut std::ffi::c_void,
        &mut t1 as *mut u32 as *mut std::ffi::c_void,
        &mut t2 as *mut u32 as *mut std::ffi::c_void,
        &mut t3 as *mut u32 as *mut std::ffi::c_void,
    ];
    unsafe {
        cudarc::driver::result::ctx::set_current(st.ctx).expect("ctx");
        let status = cudarc::driver::sys::cuLaunchKernel(
            oxide::kernel_function(&pk),
            1, 1, 1, 256, 1, 1, 0,
            cube_stream,
            pargs2.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        println!("[probe] prng on cubecl-raw stream: {status:?}");
    }
    // 用随机大数种子（模拟 next_seeds 的值域）手动 launch
    let mut r0 = 0x9e3779b9u32;
    let mut r1 = 0x7f4a7c15u32;
    let mut r2 = 0xbf58476du32;
    let mut r3 = 0x94d049bbu32;
    let mut pargs3: [*mut std::ffi::c_void; 8] = [
        &mut po2 as *mut _ as *mut std::ffi::c_void,
        &mut threads2 as *mut u32 as *mut std::ffi::c_void,
        &mut mean2 as *mut f32 as *mut std::ffi::c_void,
        &mut std2 as *mut f32 as *mut std::ffi::c_void,
        &mut r0 as *mut u32 as *mut std::ffi::c_void,
        &mut r1 as *mut u32 as *mut std::ffi::c_void,
        &mut r2 as *mut u32 as *mut std::ffi::c_void,
        &mut r3 as *mut u32 as *mut std::ffi::c_void,
    ];
    unsafe {
        cudarc::driver::result::ctx::set_current(st.ctx).expect("ctx");
        let status = cudarc::driver::sys::cuLaunchKernel(
            oxide::kernel_function(&pk),
            1, 1, 1, 256, 1, 1, 0,
            cube_stream,
            pargs3.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        println!("[probe] prng random-seeds manual: {status:?}");
    }
    // 用 oxide::load_kernel + oxide::launch 组合（完全 oxide 路径）
    let pk2 = oxide::load_kernel(&device, &prng_ptx, "prng_normal_half").expect("oxide load_kernel");
    let mut pargs4: [*mut std::ffi::c_void; 8] = [
        &mut po2 as *mut _ as *mut std::ffi::c_void,
        &mut threads2 as *mut u32 as *mut std::ffi::c_void,
        &mut mean2 as *mut f32 as *mut std::ffi::c_void,
        &mut std2 as *mut f32 as *mut std::ffi::c_void,
        &mut r0 as *mut u32 as *mut std::ffi::c_void,
        &mut r1 as *mut u32 as *mut std::ffi::c_void,
        &mut r2 as *mut u32 as *mut std::ffi::c_void,
        &mut r3 as *mut u32 as *mut std::ffi::c_void,
    ];
    match unsafe { oxide::launch(&pk2, &device, 1, 1, 1, 256, 1, 1, 0, &mut pargs4) } {
        Ok(()) => println!("[probe] oxide::load_kernel+launch OK"),
        Err(e) => println!("[probe] oxide::load_kernel+launch 失败: {e}"),
    }
    // 手动 launch 用 pk2（oxide 刚加载的 module）——隔离 module 实例 vs launch 封装
    unsafe {
        cudarc::driver::result::ctx::set_current(st.ctx).expect("ctx");
        let status = cudarc::driver::sys::cuLaunchKernel(
            oxide::kernel_function(&pk2),
            1, 1, 1, 256, 1, 1, 0,
            cube_stream,
            pargs4.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        println!("[probe] manual launch with pk2 (oxide-loaded): {status:?}");
    }
    return;
    let sum: f32 = c.clone().sum().into_scalar();
    let expected: f32 = (a.clone() + b.clone()).sum().into_scalar();
    println!("[probe] vecadd sum = {sum:.3} (期望 {expected:.3})");
}
