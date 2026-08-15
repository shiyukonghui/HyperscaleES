//! cuda-oxide 鍐呮牳鐨勫涓讳晶鍔犺浇/鍚姩灏佽锛堜粎 `gpu` feature锛夈€?//!
//! 闆嗘垚璺緞锛堣 `docs/cuda_oxide_integration_plan.md`锛夛細
//! 鐢?cuda-oxide 缂栬瘧鍣ㄦ妸 `hyperscalees-kernels` 鐨?Rust 鍐呮牳缂栬瘧涓?PTX 鏂囨湰锛?//! 瀹夸富缁?CUDA driver API 鍔犺浇锛坄cuModuleLoadData`锛夊苟鍚姩锛坄cuLaunchKernel`锛夈€?//!
//! 涓?[`crate::cublas`] 鍚屼竴濂楁満鍒讹細module/function 鎸傚湪 cubecl 鐨?context 涓?//! 锛堝鐢?`cublas::state` 鐨?ctx锛夛紝鍚姩缁戝埌 cubecl 鐨勫師濮?stream
//! 锛坄raw_stream`锛夆啋 涓?burn 绠楀瓙澶╃劧鍚屾祦鏈夊簭銆侀浂鍚屾锛堟帰閽堝疄娴嬪彲鐢級銆?//!
//! 鍐呮牳鍙傛暟鐩存帴浼?burn 寮犻噺鐨勫師濮嬭澶囨寚閽堬紙`cublas::raw_ptr` 鍚屾 resolve
//! 鏈哄埗锛変笌鏄惧紡鏍囬噺锛坘ernelParams 鎸囧悜鍙傛暟鍊硷紝瑙?[`launch`]锛夈€?
use std::ffi::c_void;

use burn::backend::cuda::CudaDevice;
use cubecl::cuda::CudaRuntime;
use cubecl::device::Device;
use cubecl::device_handle::DeviceHandle;
use cubecl::stream_id::StreamId;
use cubecl::Runtime;

use crate::cublas::state as cublas_state;

/// 鏈嶅姟鍣ㄧ被鍨嬶紙vendored cubecl-cuda 鐨?`CudaServer`锛夈€?type Server = <CudaRuntime as Runtime>::Server;

/// 宸插姞杞界殑 cuda-oxide 鍐呮牳锛坢odule + function 鍙ユ焺锛夈€?pub struct OxideKernel {
    module: cudarc::driver::sys::CUmodule,
    function: cudarc::driver::sys::CUfunction,
}

/// 渚涙帰閽?闆嗘垚浠ｇ爜璁块棶 function 鍙ユ焺銆?pub fn kernel_function(kernel: &OxideKernel) -> cudarc::driver::sys::CUfunction {
    kernel.function
}

// 鍙ユ焺鍙湪鏈ā鍧楀唴鎸夊簭浣跨敤锛屼笉璺ㄧ嚎绋嬪叡浜彲鍙樿闂紙涓?CublasState 鐩稿悓绾﹀畾锛夈€?unsafe impl Send for OxideKernel {}
unsafe impl Sync for OxideKernel {}

impl Drop for OxideKernel {
    fn drop(&mut self) {
        // 閲婃斁 module锛坒unction 鍙ユ焺闅?module 澶辨晥锛屾棤闇€鍗曠嫭閲婃斁锛夈€?        unsafe {
            cudarc::driver::sys::cuModuleUnload(self.module);
        }
    }
}

/// 浠?PTX 鏂囨湰鍔犺浇涓€涓唴鏍稿嚱鏁般€?///
/// `ptx` 涓?cuda-oxide 缂栬瘧浜у嚭鐨?PTX 鏂囨湰瀛楄妭锛?*蹇呴』浠?NUL 缁撳熬**锛?/// `cuModuleLoadData` 瑕佹眰锛夛紱`kernel_name` 涓?PTX 鍐呯殑鍑芥暟鍚嶃€?/// 鍔犺浇鍚庡唴鏍镐笌 cubecl 鍏变韩鍚屼竴 context锛堝鐢?cuBLAS 鐨?context锛夈€?pub fn load_kernel(
    device: &CudaDevice,
    ptx: &[u8],
    kernel_name: &str,
) -> Result<OxideKernel, String> {
    let st = cublas_state(device);
    unsafe {
        cudarc::driver::result::ctx::set_current(st.ctx)
            .map_err(|e| format!("璁剧疆 CUDA 涓婁笅鏂囧け璐? {e}"))?;

        let mut module = std::mem::MaybeUninit::<cudarc::driver::sys::CUmodule>::uninit();
        let status = cudarc::driver::sys::cuModuleLoadData(
            module.as_mut_ptr(),
            ptx.as_ptr() as *const c_void,
        );
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuModuleLoadData 澶辫触: {status:?}"));
        }
        let module = module.assume_init();

        let mut function = std::mem::MaybeUninit::<cudarc::driver::sys::CUfunction>::uninit();
        let name =
            std::ffi::CString::new(kernel_name).map_err(|_| "鍐呮牳鍚嶅惈 NUL".to_string())?;
        let status = cudarc::driver::sys::cuModuleGetFunction(
            function.as_mut_ptr(),
            module,
            name.as_ptr(),
        );
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            let _ = cudarc::driver::sys::cuModuleUnload(module);
            return Err(format!("cuModuleGetFunction({kernel_name}) 澶辫触: {status:?}"));
        }
        Ok(OxideKernel {
            module,
            function: function.assume_init(),
        })
    }
}

/// 鍚姩鍐呮牳锛堢粦鍒?cubecl 涓?stream锛岄浂鍚屾锛涗笌 cuBLAS 璋冪敤鍚屽簭锛夈€?///
/// `args`锛氬唴鏍稿弬鏁扮殑鎸囬拡鏁扮粍鈥斺€旀瘡涓厓绱犳槸鎸囧悜**鍙傛暟鍊?*鐨勬寚閽?/// 锛坄&mut arg as *mut _ as *mut c_void`锛夛紝涓?`cuLaunchKernel` 鐨?/// `kernelParams` 绾﹀畾涓€鑷淬€傝皟鐢ㄦ柟淇濊瘉鍙傛暟甯冨眬涓?PTX 鍐呮牳绛惧悕鍖归厤銆?/// 娉ㄦ剰锛氬弬鏁板€煎繀椤荤敱**鍏峰悕鍙橀噺**鎵胯浇骞跺瓨娲诲埌 launch 杩斿洖锛堜复鏃跺€间細鎮瀭锛夈€?///
/// # Safety
/// 鍙傛暟鎸囬拡蹇呴』鎸囧悜涓庡唴鏍哥鍚嶅尮閰嶇殑鍊间笖鍦ㄥ唴鏍告墽琛屾湡闂存湁鏁堛€?pub unsafe fn launch(
    kernel: &OxideKernel,
    device: &CudaDevice,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_mem_bytes: u32,
    args: &mut [*mut c_void],
) -> Result<(), String> {
    let st = cublas_state(device);

    // 鐩存帴 launch 鍒?cubecl 涓绘祦锛堝悓娴佹湁搴忋€侀浂鍚屾锛涗笌 cuBLAS 闆嗘垚鍚屾満鍒讹級銆?    // 娉ㄦ剰锛歝uLaunchKernel 鍙傛暟椤哄簭鏄?(..., hStream, kernelParams, extra)鈥斺€?    // kernelParams 浼?args 鏁扮粍銆乪xtra 浼?null锛堟浘鎶婁袱鑰呭啓鍙嶅鑷?    // CUDA_ERROR_INVALID_VALUE锛屾帰閽堥€愬瓧鑺傚姣旀墠瀹氫綅锛夈€?    let dh = DeviceHandle::<Server>::new(device.to_id());
    let stream = dh
        .submit_blocking(|s| s.raw_stream(StreamId::current()) as usize)
        .expect("鍙?CUDA stream 澶辫触") as *mut cudarc::driver::sys::CUstream_st;
    cudarc::driver::result::ctx::set_current(st.ctx)
        .map_err(|e| format!("璁剧疆 CUDA 涓婁笅鏂囧け璐? {e}"))?;

    let status = cudarc::driver::sys::cuLaunchKernel(
        kernel.function,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared_mem_bytes,
        stream,
        args.as_mut_ptr(),    // kernelParams
        std::ptr::null_mut(), // extra
    );
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuLaunchKernel 澶辫触: {status:?}"));
    }
    Ok(())
}

/// 鍦ㄦ寚瀹氭祦涓婂惎鍔ㄥ唴鏍革紙鍐呴儴璺緞锛涗緵鎺㈤拡/鐗规畩鍦烘櫙澶嶇敤锛夈€?pub unsafe fn launch_on_stream(
    kernel: &OxideKernel,
    device: &CudaDevice,
    stream: *mut cudarc::driver::sys::CUstream_st,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_mem_bytes: u32,
    args: &mut [*mut c_void],
) -> Result<(), String> {
    let st = cublas_state(device);
    cudarc::driver::result::ctx::set_current(st.ctx)
        .map_err(|e| format!("璁剧疆 CUDA 涓婁笅鏂囧け璐? {e}"))?;
    let status = cudarc::driver::sys::cuLaunchKernel(
        kernel.function,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared_mem_bytes,
        stream,
        args.as_mut_ptr(),    // kernelParams
        std::ptr::null_mut(), // extra
    );
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuLaunchKernel 澶辫触: {status:?}"));
    }
    Ok(())
}

// ===========================================================================
// 鍏蜂綋鍐呮牳锛氬崐閲忔鎬佸櫔澹扮敓鎴愶紙cuda-oxide 缂栬瘧锛孭TX 鍐呭祵锛?// ===========================================================================

/// PRNG 鍐呮牳 PTX锛坈uda-oxide 缂栬瘧锛歚snn_prng` 绀轰緥 鈫?llvm-link libdevice 鈫?/// opt 瑁佸壀 鈫?llc锛岃 `docs/cuda_oxide_integration_plan.md`锛夈€?/// 娉ㄦ剰锛歝uModuleLoadData 瑕佹眰 PTX 鏂囨湰浠?NUL 缁撳熬锛屾晠鐢?include_str + "\0"銆?const PRNG_PTX: &[u8] =
    concat!(include_str!("../../hyperscalees-kernels/ptx/prng_normal_half.ptx"), "\0").as_bytes();
/// 姣忕嚎绋嬪厓绱犳暟锛堜笌鍐呮牳甯搁噺 ELEMS_PER_THREAD 涓€鑷达級銆?const PRNG_ELEMS_PER_THREAD: usize = 128;

/// 宸插姞杞界殑 PRNG 鍐呮牳锛堣繘绋嬬骇缂撳瓨锛屽姞杞戒竴娆★級銆?fn prng_kernel(device: &CudaDevice) -> &'static OxideKernel {
    static KERNEL: std::sync::OnceLock<OxideKernel> = std::sync::OnceLock::new();
    KERNEL.get_or_init(|| {
        load_kernel(device, PRNG_PTX, "prng_normal_half").expect("鍔犺浇 prng_normal_half PTX 澶辫触")
    })
}

/// 鍐呮牳鍙傛暟绉嶅瓙锛堟瘡娆¤皟鐢ㄨ嚜澧?+ 鏃堕棿娣峰悎锛屼繚璇佹瘡娆＄敓鎴愪笉鍚屽簭鍒楋級銆?fn next_seeds() -> [u32; 4] {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);
    let c = CTR.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0) as u64;
    let mix = |x: u64| -> u32 {
        let mut h = x;
        h ^= h >> 30;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        h as u32
    };
    [mix(c), mix(c >> 32), mix(t), mix(t >> 32)]
}

/// 鍗婇噺姝ｆ€佸～鍏咃細`out` (n/2, r, b) 杩炵画寮犻噺锛屽～鍏?`N(mean, std虏)`銆?///
/// 涓庤缁冪儹璺緞鐨勩€屽崐鍣０銆嶇害瀹氶厤濂楋紙閰嶅鐢辨秷璐规柟闅愬惈鏂藉姞锛夈€傚唴鏍镐负
/// cuda-oxide 缂栬瘧鐨?PTX锛岀粡 cudarc 鍦?cubecl 涓绘祦涓婂惎鍔紙鍚屾祦鏈夊簭锛岄浂鍚屾锛夈€?pub fn prng_normal_half(
    out: &burn::tensor::Tensor<crate::B, 3>,
    mean: f32,
    std: f32,
    device: &CudaDevice,
) -> Result<(), String> {
    let n_elems = out.shape().dims::<3>().iter().product::<usize>();
    debug_assert_eq!(n_elems % PRNG_ELEMS_PER_THREAD, 0, "鍏冪礌鏁伴渶涓?128 鐨勬暣鏁板€?);
    let total_threads = (n_elems / PRNG_ELEMS_PER_THREAD) as u32;

    let cube = crate::cublas::as_cube(out);
    let ptr = crate::cublas::raw_ptr(&cube, device) as *mut f32;

    let seeds = next_seeds();
    // cuLaunchKernel 瑕佹眰 kernelParams 鎸囧悜鐨?*鍙傛暟鍊兼寜 8 瀛楄妭瀵归綈**锛坲32/f32
    // 鏅€氭爤鍙橀噺鍙繚璇?4 瀛楄妭瀵归綈锛屼細闅忔満瑙﹀彂 CUDA_ERROR_INVALID_VALUE锛夛紱
    // 鐢?repr(C, align(8)) 鍖呰姣忎釜鍙傛暟鍊硷紝鍐嶅彇鍐呴儴瀛楁鐨勬寚閽堛€?    #[repr(C, align(8))]
    struct A8<T>(T);
    let mut arg_ptr = A8(ptr as *mut c_void);
    let mut arg_threads = A8(total_threads);
    let mut arg_mean = A8(mean);
    let mut arg_std = A8(std);
    let mut arg_s0 = A8(seeds[0]);
    let mut arg_s1 = A8(seeds[1]);
    let mut arg_s2 = A8(seeds[2]);
    let mut arg_s3 = A8(seeds[3]);
    let mut args: [*mut c_void; 8] = [
        &mut arg_ptr.0 as *mut *mut c_void as *mut c_void,
        &mut arg_threads.0 as *mut u32 as *mut c_void,
        &mut arg_mean.0 as *mut f32 as *mut c_void,
        &mut arg_std.0 as *mut f32 as *mut c_void,
        &mut arg_s0.0 as *mut u32 as *mut c_void,
        &mut arg_s1.0 as *mut u32 as *mut c_void,
        &mut arg_s2.0 as *mut u32 as *mut c_void,
        &mut arg_s3.0 as *mut u32 as *mut c_void,
    ];

    const BLOCK: u32 = 256;
    let grid = total_threads.div_ceil(BLOCK);
    let kernel = prng_kernel(device);
    unsafe { launch(kernel, device, grid, 1, 1, BLOCK, 1, 1, 0, &mut args) }
}

// ===========================================================================
// 鍏蜂綋鍐呮牳锛氶厤瀵瑰悎骞?einsum锛堣瀺鍚堥澶勭悊锛宑uda-oxide 缂栬瘧锛孭TX 鍐呭祵锛?// ===========================================================================

/// einsum 鍐呮牳 PTX锛坈uda-oxide 缂栬瘧锛歚snn_einsum` 绀轰緥 鈫?llvm-link libdevice 鈫?/// opt 瑁佸壀 鈫?llc锛岃 `docs/cuda_oxide_integration_plan.md`锛夈€?const EINSUM_PTX: &[u8] = concat!(
    include_str!("../../hyperscalees-kernels/ptx/einsum_pair_fused.ptx"),
    "\0"
)
.as_bytes();

/// 宸插姞杞界殑 einsum 鍐呮牳锛堣繘绋嬬骇缂撳瓨锛屽姞杞戒竴娆★級銆?fn einsum_kernel(device: &CudaDevice) -> &'static OxideKernel {
    static KERNEL: std::sync::OnceLock<OxideKernel> = std::sync::OnceLock::new();
    KERNEL.get_or_init(|| {
        let k = load_kernel(device, EINSUM_PTX, "einsum_pair_fused")
            .expect("鍔犺浇 einsum_pair_fused PTX 澶辫触");
        if std::env::var("DEBUG_OXIDE").map(|v| v == "1").unwrap_or(false) {
            // 鏌ヨ ptxas 瀹為檯鍒嗛厤锛氬瘎瀛樺櫒鏁颁笌 local memory锛堟孩鍑猴級瀛楄妭鏁般€?            let mut regs: i32 = 0;
            let mut local: i32 = 0;
            unsafe {
                cudarc::driver::sys::cuFuncGetAttribute(
                    &mut regs,
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
                    k.function,
                );
                cudarc::driver::sys::cuFuncGetAttribute(
                    &mut local,
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                    k.function,
                );
            }
            eprintln!("[oxide] einsum_pair_fused: numRegs={regs} localSizeBytes={local}");
        }
        k
    })
}

/// 铻嶅悎閰嶅 einsum锛坈uda-oxide 鍐呮牳锛夛細涓?`cublas::lora_einsum_pair_cublas`
/// 鏁板涓€鑷达紙`g_raw = 危_i (f_i+f_{half+i})路A'_i鈯桞'_i`锛宍g_ones = 2路危_i A'_i鈯桞'_i`锛夛紝
/// 浣嗘妸銆宻lice + f_pair 鍔犳潈 + cat 鎷兼帴銆嶅叏閮ㄨ瀺鍚堣繘鍏变韩鍐呭瓨鍔犺浇锛孉/B 鍚勫彧璇讳竴閬嶏紱
/// 杈撳嚭缁?f32 鍘熷瓙绱姞锛坰plit-K 鍚堝苟锛夈€?///
/// 瑕佹眰锛歚a_t`/`b_t` 琛屼富搴忚繛缁紙dim2 stride 1锛宒im0/dim1 stride 鏄惧紡浼犲叆锛?/// 鏀寔 burn 鐨?16 瀛楄妭琛屽榻?pitch锛夛紱`2a 鈮?256`锛涜緭鍑?`(a, b)` 鍏堢疆闆躲€?pub fn einsum_pair_fused(
    a_t: &burn::tensor::Tensor<crate::B, 3>,    // (half, r, a) 鍗婇噺 A' 鍣０
    b_t: &burn::tensor::Tensor<crate::B, 3>,    // (half, r, b) 鍗婇噺 B' 鍣０
    scores: &burn::tensor::Tensor<crate::B, 1>, // (n,) 鍘熷鍒嗘暟
    device: &CudaDevice,
) -> Result<(burn::tensor::Tensor<crate::B, 2>, burn::tensor::Tensor<crate::B, 2>), String> {
    use crate::cublas::{as_cube, raw_ptr};
    let [half, r, a] = a_t.dims();
    let [_, _, b] = b_t.dims();
    assert!(half * r > 0 && 2 * a <= 256, "einsum 鍐呮牳瑕佹眰 0 < 2a 鈮?256锛屽疄闄?2a={}", 2 * a);
    assert_eq!(b_t.dims(), [half, r, b], "b_t 褰㈢姸涓嶅尮閰?);

    let ca = as_cube(a_t);
    let cb = as_cube(b_t);
    let cs = as_cube(scores);
    let s_a = ca.meta.strides();
    let s_b = cb.meta.strides();
    assert_eq!(
        (s_a[2], s_b[2]),
        (1, 1),
        "einsum 鍐呮牳瑕佹眰琛屼富搴忚繛缁緭鍏ワ紙innermost stride 1锛夛紝瀹為檯 {:?}/{:?}",
        s_a,
        s_b
    );

    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let ps = raw_ptr(&cs, device);
    let k_total = half * r;

    // 杈撳嚭锛堣皟鐢ㄦ柟璇箟 = 鍘熷瓙绱姞锛屽繀椤婚浂鍒濆鍖栵級銆?    let g_raw: burn::tensor::Tensor<crate::B, 2> = burn::tensor::Tensor::zeros([a, b], device);
    let g_ones: burn::tensor::Tensor<crate::B, 2> = burn::tensor::Tensor::zeros([a, b], device);
    let cg_raw = as_cube(&g_raw);
    let cg_ones = as_cube(&g_ones);
    let pgr = raw_ptr(&cg_raw, device);
    let pgo = raw_ptr(&cg_ones, device);

    // split-K锛欿=384000 鈫?kslice=4000锛?25 涓?BK=32 鍧楋紝鏁撮櫎锛夛紱N=112 涓€鍒楀潡銆?    let k_slices = std::env::var("OXIDE_KS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| std::cmp::max(1, k_total.div_ceil(4000))) as u32;
    let n_tiles = (b as u32).div_ceil(112);

    if std::env::var("DEBUG_OXIDE").map(|v| v == "1").unwrap_or(false) {
        eprintln!(
            "[oxide] einsum: a({:?}) strides={:?} b({:?}) strides={:?} scores({:?}) \
             half={} r={} a={} b={} K={} k_slices={} n_tiles={}",
            a_t.dims(),
            s_a,
            b_t.dims(),
            s_b,
            scores.dims(),
            half,
            r,
            a,
            b,
            k_total,
            k_slices,
            n_tiles
        );
    }

    let kernel = einsum_kernel(device);
    // cuLaunchKernel 鍙傛暟鍊奸渶 8 瀛楄妭瀵归綈锛堣 prng_normal_half 娉ㄩ噴锛夈€?    #[repr(C, align(8))]
    struct A8<T>(T);
    // 杈撳嚭寮犻噺鐨勮 stride锛坆urn 256B 瀵归綈 pitch锛屽唴鏍稿師瀛愬啓鍦板潃鐢級
    let g_s_raw = cg_raw.meta.strides()[0] as u32;
    let g_s_ones = cg_ones.meta.strides()[0] as u32;
    debug_assert_eq!(g_s_raw, g_s_ones, "杈撳嚭寮犻噺 stride 搴斾竴鑷?);
    let mut arg_a = A8(pa as *mut c_void);
    let mut arg_a0 = A8(s_a[0] as u32);
    let mut arg_a1 = A8(s_a[1] as u32);
    let mut arg_ad = A8(a as u32);
    let mut arg_b = A8(pb as *mut c_void);
    let mut arg_b0 = A8(s_b[0] as u32);
    let mut arg_b1 = A8(s_b[1] as u32);
    let mut arg_bd = A8(b as u32);
    let mut arg_s = A8(ps as *mut c_void);
    let mut arg_h = A8(half as u32);
    let mut arg_r = A8(r as u32);
    let mut arg_gr = A8(pgr as *mut c_void);
    let mut arg_go = A8(pgo as *mut c_void);
    let mut arg_gs = A8(g_s_raw);
    let mut arg_k = A8(k_total as u32);
    let mut arg_ks = A8(k_slices);
    let mut args: [*mut c_void; 16] = [
        &mut arg_a.0 as *mut *mut c_void as *mut c_void,
        &mut arg_a0.0 as *mut u32 as *mut c_void,
        &mut arg_a1.0 as *mut u32 as *mut c_void,
        &mut arg_ad.0 as *mut u32 as *mut c_void,
        &mut arg_b.0 as *mut *mut c_void as *mut c_void,
        &mut arg_b0.0 as *mut u32 as *mut c_void,
        &mut arg_b1.0 as *mut u32 as *mut c_void,
        &mut arg_bd.0 as *mut u32 as *mut c_void,
        &mut arg_s.0 as *mut *mut c_void as *mut c_void,
        &mut arg_h.0 as *mut u32 as *mut c_void,
        &mut arg_r.0 as *mut u32 as *mut c_void,
        &mut arg_gr.0 as *mut *mut c_void as *mut c_void,
        &mut arg_go.0 as *mut *mut c_void as *mut c_void,
        &mut arg_gs.0 as *mut u32 as *mut c_void,
        &mut arg_k.0 as *mut u32 as *mut c_void,
        &mut arg_ks.0 as *mut u32 as *mut c_void,
    ];
    unsafe { launch(kernel, device, n_tiles, k_slices, 1, 512, 1, 1, 0, &mut args) }?;
    if std::env::var("DEBUG_OXIDE").map(|v| v == "1").unwrap_or(false) {
        unsafe {
            let rd = |p: *mut c_void| *(p as *const u64);
            let ru = |p: *mut c_void| *(p as *const u32);
            eprintln!(
                "[oxide] args: a={:#x} a_s0={} a_s1={} a={} b={:#x} b_s0={} b_s1={} b={} \
                 scores={:#x} half={} r={} g_raw={:#x} g_ones={:#x} K={} k_slices={} grid=({}, {})",
                rd(args[0]),
                ru(args[1]),
                ru(args[2]),
                ru(args[3]),
                rd(args[4]),
                ru(args[5]),
                ru(args[6]),
                ru(args[7]),
                rd(args[8]),
                ru(args[9]),
                ru(args[10]),
                rd(args[11]),
                rd(args[12]),
                ru(args[13]),
                ru(args[14]),
                n_tiles,
                k_slices
            );
        }
    }
    Ok((g_raw, g_ones))
}

/// 璇婃柇灏佽锛歞ump 棣栦釜 chunk 鐨?acc 妲斤紙tx<2 绾跨▼锛夛紝杩斿洖 CPU 鎷疯礉銆?pub fn einsum_dump_acc(
    a_t: &burn::tensor::Tensor<crate::B, 3>,
    b_t: &burn::tensor::Tensor<crate::B, 3>,
    scores: &burn::tensor::Tensor<crate::B, 1>,
    device: &CudaDevice,
) -> Result<Vec<f32>, String> {
    use crate::cublas::{as_cube, raw_ptr};
    let [half, r, a] = a_t.dims();
    let [_, _, b] = b_t.dims();
    let ca = as_cube(a_t);
    let cb = as_cube(b_t);
    let cs = as_cube(scores);
    let s_a = ca.meta.strides();
    let s_b = cb.meta.strides();
    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let ps = raw_ptr(&cs, device);
    let k_total = half * r;

    let ad: burn::tensor::Tensor<crate::B, 1> =
        burn::tensor::Tensor::zeros([2 * 16 * 112], device);
    let cad = as_cube(&ad);
    let pad = raw_ptr(&cad, device);

    let kernel = load_kernel(device, EINSUM_PTX, "einsum_pair_dump_acc")
        .expect("鍔犺浇 einsum_pair_dump_acc PTX 澶辫触");
    #[repr(C, align(8))]
    struct A8<T>(T);
    let mut arg_a = A8(pa as *mut c_void);
    let mut arg_a0 = A8(s_a[0] as u32);
    let mut arg_a1 = A8(s_a[1] as u32);
    let mut arg_ad = A8(a as u32);
    let mut arg_b = A8(pb as *mut c_void);
    let mut arg_b0 = A8(s_b[0] as u32);
    let mut arg_b1 = A8(s_b[1] as u32);
    let mut arg_bd = A8(b as u32);
    let mut arg_s = A8(ps as *mut c_void);
    let mut arg_h = A8(half as u32);
    let mut arg_r = A8(r as u32);
    let mut arg_pa = A8(pad as *mut c_void);
    let mut arg_k = A8(k_total as u32);
    let mut arg_ks = A8(1u32);
    // dump_acc 鏈?14 鍙傛暟锛堟棤 b_dump锛夆€斺€旂 12 涓槸 acc_dump锛?3/14 鏄?k_total/k_slices
    let mut args: [*mut c_void; 14] = [
        &mut arg_a.0 as *mut *mut c_void as *mut c_void,
        &mut arg_a0.0 as *mut u32 as *mut c_void,
        &mut arg_a1.0 as *mut u32 as *mut c_void,
        &mut arg_ad.0 as *mut u32 as *mut c_void,
        &mut arg_b.0 as *mut *mut c_void as *mut c_void,
        &mut arg_b0.0 as *mut u32 as *mut c_void,
        &mut arg_b1.0 as *mut u32 as *mut c_void,
        &mut arg_bd.0 as *mut u32 as *mut c_void,
        &mut arg_s.0 as *mut *mut c_void as *mut c_void,
        &mut arg_h.0 as *mut u32 as *mut c_void,
        &mut arg_r.0 as *mut u32 as *mut c_void,
        &mut arg_pa.0 as *mut *mut c_void as *mut c_void,
        &mut arg_k.0 as *mut u32 as *mut c_void,
        &mut arg_ks.0 as *mut u32 as *mut c_void,
    ];
    unsafe {
        launch(&kernel, device, 1, 1, 1, 256, 1, 1, 0, &mut args)?;
    }
    Ok(ad.into_data().into_vec::<f32>().map_err(|e| format!("{e:?}"))?)
}

/// 璇婃柇灏佽锛歞ump A_SH/B_SH 鍓?2 琛岋紙璋冭瘯鐢級锛岃繑鍥?CPU 鎷疯礉銆?pub fn einsum_dump(
    a_t: &burn::tensor::Tensor<crate::B, 3>,
    b_t: &burn::tensor::Tensor<crate::B, 3>,
    scores: &burn::tensor::Tensor<crate::B, 1>,
    device: &CudaDevice,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    use crate::cublas::{as_cube, raw_ptr};
    let [half, r, a] = a_t.dims();
    let [_, _, b] = b_t.dims();
    let ca = as_cube(a_t);
    let cb = as_cube(b_t);
    let cs = as_cube(scores);
    let s_a = ca.meta.strides();
    let s_b = cb.meta.strides();
    let pa = raw_ptr(&ca, device);
    let pb = raw_ptr(&cb, device);
    let ps = raw_ptr(&cs, device);
    let k_total = half * r;

    let ad: burn::tensor::Tensor<crate::B, 1> =
        burn::tensor::Tensor::zeros([2 * 257], device);
    let bd: burn::tensor::Tensor<crate::B, 1> =
        burn::tensor::Tensor::zeros([2 * 112], device);
    let cad = as_cube(&ad);
    let cbd = as_cube(&bd);
    let pad = raw_ptr(&cad, device);
    let pbd = raw_ptr(&cbd, device);

    let kernel = load_kernel(device, EINSUM_PTX, "einsum_pair_dump_tiles")
        .expect("鍔犺浇 einsum_pair_dump_tiles PTX 澶辫触");
    #[repr(C, align(8))]
    struct A8<T>(T);
    let mut arg_a = A8(pa as *mut c_void);
    let mut arg_a0 = A8(s_a[0] as u32);
    let mut arg_a1 = A8(s_a[1] as u32);
    let mut arg_ad = A8(a as u32);
    let mut arg_b = A8(pb as *mut c_void);
    let mut arg_b0 = A8(s_b[0] as u32);
    let mut arg_b1 = A8(s_b[1] as u32);
    let mut arg_bd = A8(b as u32);
    let mut arg_s = A8(ps as *mut c_void);
    let mut arg_h = A8(half as u32);
    let mut arg_r = A8(r as u32);
    let mut arg_pa = A8(pad as *mut c_void);
    let mut arg_pb = A8(pbd as *mut c_void);
    let mut arg_k = A8(k_total as u32);
    let mut arg_ks = A8(1u32);
    let mut args: [*mut c_void; 15] = [
        &mut arg_a.0 as *mut *mut c_void as *mut c_void,
        &mut arg_a0.0 as *mut u32 as *mut c_void,
        &mut arg_a1.0 as *mut u32 as *mut c_void,
        &mut arg_ad.0 as *mut u32 as *mut c_void,
        &mut arg_b.0 as *mut *mut c_void as *mut c_void,
        &mut arg_b0.0 as *mut u32 as *mut c_void,
        &mut arg_b1.0 as *mut u32 as *mut c_void,
        &mut arg_bd.0 as *mut u32 as *mut c_void,
        &mut arg_s.0 as *mut *mut c_void as *mut c_void,
        &mut arg_h.0 as *mut u32 as *mut c_void,
        &mut arg_r.0 as *mut u32 as *mut c_void,
        &mut arg_pa.0 as *mut *mut c_void as *mut c_void,
        &mut arg_pb.0 as *mut *mut c_void as *mut c_void,
        &mut arg_k.0 as *mut u32 as *mut c_void,
        &mut arg_ks.0 as *mut u32 as *mut c_void,
    ];
    unsafe {
        launch(
            &kernel, device, 1, 1, 1, 256, 1, 1, 0, &mut args,
        )?;
    }
    Ok((
        ad.into_data().into_vec::<f32>().map_err(|e| format!("{e:?}"))?,
        bd.into_data().into_vec::<f32>().map_err(|e| format!("{e:?}"))?,
    ))
}

#[cfg(test)]
mod tests {
    /// 鑷锛氭ā鍧楀彲缂栬瘧銆佺被鍨?绛惧悕瀛樺湪銆?    #[test]
    fn skeleton_compiles() {
        let _ = std::any::type_name::<crate::oxide::OxideKernel>();
    }
}
