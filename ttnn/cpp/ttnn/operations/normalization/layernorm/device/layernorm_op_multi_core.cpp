// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include <tt-metalium/circular_buffer_config.hpp>
#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/metal2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include <filesystem>
#include <optional>
#include <bit>
#include <vector>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// computes layernorm(a+*b)*gamma + beta
// if b is nullptr it's treated as zero (no addition)
bool CB_can_fit_in_L1(
    uint32_t in0_size,
    uint32_t in1_size,
    uint32_t out0_size,
    uint32_t im0_size,
    uint32_t im3_size,
    uint32_t in5_size,
    uint32_t in6_size,
    uint32_t im6_size,
    uint32_t im5_size,
    uint32_t im4_size,
    uint32_t im1_size,
    uint32_t in2_size,
    uint32_t in3_size,
    uint32_t im2_size,
    uint32_t recip_size,
    uint32_t in_rm_size,
    uint32_t l1_size) {
    uint32_t sum = 0;
    sum += in0_size;
    sum += in1_size;
    sum += out0_size;
    sum += im0_size;
    sum += im3_size;
    sum += in5_size;
    sum += in6_size;
    sum += im6_size;
    sum += im5_size;
    sum += im4_size;
    sum += im1_size;
    sum += in2_size;
    sum += in3_size;
    sum += im2_size;
    sum += recip_size;
    sum += in_rm_size;
    return sum < l1_size * 0.95;
}

using namespace tt::tt_metal::experimental;

// Metal 2.0 named resource handles for the interleaved layernorm ProgramSpec.
// One DFBSpecName per distinct legacy CB index. The accessor_name used at each binding
// site equals the unique_id string, so the kernel-side handle is dfb::<same name> — matching
// the variable names the legacy kernels already used (cb_in, cb_out, ...).
const DFBSpecName DFB_IN{"cb_in"};                    // c_0
const DFBSpecName DFB_INB{"cb_inb"};                  // c_1
const DFBSpecName DFB_SCALER{"cb_scaler"};            // c_2
const DFBSpecName DFB_EPS{"cb_eps"};                  // c_3
const DFBSpecName DFB_GAMMA{"cb_gamma"};              // c_5
const DFBSpecName DFB_BETA{"cb_beta"};                // c_6
const DFBSpecName DFB_OUT{"cb_out"};                  // c_16
const DFBSpecName DFB_EX{"cb_ex"};                    // c_18
const DFBSpecName DFB_EX2{"cb_ex2"};                  // c_19
const DFBSpecName DFB_XMM2{"cb_xmm2"};                // c_20
const DFBSpecName DFB_EX2PE{"cb_ex2pe"};              // c_21
const DFBSpecName DFB_FUSION{"cb_fusion"};            // c_22
const DFBSpecName DFB_X{"cb_x"};                      // c_23
const DFBSpecName DFB_XMM{"cb_xmm"};                  // c_24
const DFBSpecName DFB_RECIPROCALS{"cb_reciprocals"};  // c_25 (borrowed from recip tensor)
const DFBSpecName DFB_ACCUMULATE{"cb_accumulate"};    // c_26
const DFBSpecName DFB_IN_RM{"cb_in_rm"};              // c_27
const DFBSpecName DFB_OUT_RM{"cb_out_rm"};            // c_28
const DFBSpecName DFB_X_WELFORD{"cb_x_welford"};      // c_29 (alias of cb_in / cb_x)
const DFBSpecName DFB_EX_WELFORD{"cb_ex_welford"};    // c_30 (alias of cb_ex)
const DFBSpecName DFB_EX2_WELFORD{"cb_ex2_welford"};  // c_31 (alias of cb_ex2)

const TensorParamName PARAM_INPUT{"input"};
const TensorParamName PARAM_RESIDUAL{"residual"};
const TensorParamName PARAM_GAMMA{"gamma"};
const TensorParamName PARAM_BETA{"beta"};
const TensorParamName PARAM_OUTPUT{"output"};
const TensorParamName PARAM_RECIP{"recip"};

const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName WRITER_KERNEL{"writer"};
const KernelSpecName COMPUTE_KERNEL{"compute"};

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts LayerNormMultiCoreProgramFactory::create_program_spec(
    const LayerNormParams& operation_attributes, const LayerNormInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using namespace tt::tt_metal::experimental;

    // Extract from operation_attributes and tensor_args
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const auto& gamma = tensor_args.weight;
    const auto& beta = tensor_args.bias;
    auto& output = tensor_return_value;
    bool rms_norm = operation_attributes.norm_type == LayerNormType::RMSNORM;
    float eps = operation_attributes.eps;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    // Extract program config
    bool legacy_reduction = false;
    bool legacy_rsqrt = false;
    bool use_welford = false;
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, LayerNormDefaultProgramConfig>) {
                legacy_reduction = program_config.legacy_reduction;
                legacy_rsqrt = program_config.legacy_rsqrt;
                use_welford = program_config.use_welford;
            }
        },
        operation_attributes.program_config);

    const auto& logical_shape = a.logical_shape();
    const auto& padded_shape = a.padded_shape();
    uint32_t W = logical_shape[-1];
    const bool input_is_row_major = a.layout() == Layout::ROW_MAJOR;
    uint32_t Wp = padded_shape[-1], Hp = padded_shape[-2];
    uint32_t HWp = Hp * Wp;
    uint32_t NC = a.physical_volume() / HWp;
    // For ROW_MAJOR inputs the tensor height is not padded to tile boundaries.
    // Round Hp up to the next TILE_HEIGHT multiple so that Ht >= 1 for any H < TILE_HEIGHT.
    // HWp and NC are computed above from the original (unpadded) Hp, so they remain correct.
    if (input_is_row_major) {
        Hp = tt::round_up(Hp, TILE_HEIGHT);
    }
    // Total logical (non-padded) row count. Used by RM reader/writer kernels to
    // avoid OOB DRAM reads/writes when H is not a multiple of TILE_HEIGHT.
    const uint32_t H_logical = static_cast<uint32_t>(NC) * static_cast<uint32_t>(logical_shape[-2]);

    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute

    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    //////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    IDevice* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                Circular Buffer Data Format Setup
    //////////////////////////////////////////////////////////////////////////
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    // Data span in tiles, rounded up to tile boundaries
    uint32_t Wt = Wp / tile_width;
    uint32_t Ht = Hp / tile_height;

    // Block size that maximizes dest usage depending on
    // whether fp32 accumulation is enabled
    uint32_t block_size = fp32_dest_acc_en ? 4 : 8;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format = gamma.has_value()
                                              ? tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype())
                                              : tt::DataFormat::Float16_b;
    tt::DataFormat beta_cb_data_format = beta.has_value()
                                             ? tt::tt_metal::datatype_to_dataformat_converter(beta.value().dtype())
                                             : tt::DataFormat::Float16_b;
    tt::DataFormat reciprocal_cb_data_format = tt::DataFormat::Float32;

    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    tt::DataFormat scaler_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scaler_tile_size = tt::tile_size(scaler_cb_data_format);
    uint32_t gamma_single_tile_size = tt::tile_size(gamma_cb_data_format);
    uint32_t beta_single_tile_size = tt::tile_size(beta_cb_data_format);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "gamma_cb_data_format: {}", gamma_cb_data_format);
    log_debug(tt::LogOp, "beta_cb_data_format: {}", beta_cb_data_format);
    log_debug(tt::LogOp, "reciprocal_cb_data_format: {}", reciprocal_cb_data_format);
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);

    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;
    if (b) {
        inb_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.value().dtype());
        inb_single_tile_size = tt::tile_size(inb_data_format);
    }

    // (Legacy buffer-address extractions for a / b / gamma / beta / output removed: those addresses
    //  now flow through TensorBindings, not runtime args.)

    uint32_t num_tile_rows = NC * Ht;

    // Production always used the default core range; the core_range_set parameter existed only
    // for the (now-removed) create_descriptor pybind hook. Inline the production default.
    CoreRangeSet requested_cores = default_core_range(device);

    // Use split_work_to_cores to properly distribute tile rows across available cores
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = split_work_to_cores(requested_cores, num_tile_rows, true /* row_wise */);

    // Use passed-in reciprocal LUT tensor if using Welford
    std::optional<Tensor> recip_tensor = std::nullopt;
    uint32_t reciprocal_CB_size_bytes = 0;
    if (use_welford) {
        TT_FATAL(tensor_args.recip_tensor.has_value(), "Reciprocal tensor not provided for Welford layernorm");
        recip_tensor = tensor_args.recip_tensor;
        reciprocal_CB_size_bytes = recip_tensor->buffer()->aligned_size_per_bank();
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    auto use_row_major_kernel = (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) or
                                (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR);
    // Size the small-kernel CBs to be a multiple of the block size
    uint32_t Wt_next_block_up = tt::round_up(Wt, block_size);
    uint32_t in0_t =
        Wt_next_block_up;  // cb_x for no pre-add variant, x=a+b for fused pre-add, extra space for some buffering
    uint32_t in1_t = block_size * 2;  // buffer for fused pre-add b tensor
    uint32_t out0_t = input_is_row_major ? Wt_next_block_up : block_size * 2;
    uint32_t im0_t = Wt_next_block_up;  // buffer for saving xmm
    uint32_t im3_t = Wt_next_block_up;  // buffer for xmm^2
    uint32_t in5_t = Wt_next_block_up;  // buffer for gamma
    uint32_t in6_t = Wt_next_block_up;  // buffer for beta
    uint32_t im6_t = block_size * 2;  // x=a+b reuse for x-E[x] computation plus a bit extra for buffering
    if (b) {
        im6_t = Wt_next_block_up;
        in0_t = 2 * block_size;
    }
    uint32_t im5_t = 2 * block_size;  // for buffering to/from *gamma/+beta
    uint32_t im4_t = 8;               // 8 just in case, 4 would prob suffice
    uint32_t im1_t = 2;
    uint32_t in2_t = 2;  // scaler for reduce coming from reader
    uint32_t in3_t = 2;  // epsilon coming from reader
    uint32_t im2_t = 2;  //

    bool large_tensor_needed = false;
    // The following constants were chosen empirically to
    // maximize the buffer size while still fitting the
    // largest cases (fused pre-add + gamma + beta) in L1.
    // There is room for optimization here based on different
    // conditions (like what buffers are actually used),
    // but having two constants for all cases is simpler.
    //
    // The base values (56 / 112) are calibrated for bfloat16 intermediate tiles.
    // When fp32_dest_acc_en is true, cb_data_format is Float32 so single_tile_size
    // doubles (4096 B vs 2048 B). All intermediate CBs (im0_t, im3_t, im5_t, …) use
    // single_tile_size, so the same number of tiles takes 2x the L1 space.  Scaling
    // by bfloat16_tile_size / single_tile_size keeps the total intermediate CB
    // footprint within the empirically validated L1 budget regardless of the
    // accumulation data format.
    const uint32_t with_weights_max_size = 56 * bfloat16_tile_size / single_tile_size;
    const uint32_t without_weights_max_size = 112 * bfloat16_tile_size / single_tile_size;
    // cb_in_rm (CB 27): double-buffered staging for in-flight tilization.
    // Two blocks let the reader DMA the next block from DRAM while compute tilizes the current
    // block, hiding DRAM read latency.  Only allocated when input_is_row_major.
    const uint32_t in_rm_tiles = input_is_row_major ? 2 * block_size : 0;
    const uint32_t in_rm_size = in_rm_tiles * in_single_tile_size;
    // cb_out_rm (CB 28): double-buffered staging for the RM writer.
    // Two blocks let the writer drain the previous block to DRAM while compute untilizes the
    // next block into the CB, hiding DRAM write latency.  Only allocated when input_is_row_major.
    const uint32_t out_rm_tiles = input_is_row_major ? 2 * block_size : 0;
    const uint32_t out_rm_size = out_rm_tiles * out_single_tile_size;

    bool cb_fits_in_L1 = CB_can_fit_in_L1(
        in0_t * in_single_tile_size,
        in1_t * inb_single_tile_size,
        out0_t * out_single_tile_size,
        im0_t * single_tile_size,
        im3_t * single_tile_size,
        in5_t * gamma_single_tile_size,
        in6_t * beta_single_tile_size,
        im6_t * single_tile_size,
        im5_t * single_tile_size,
        im4_t * single_tile_size,
        im1_t * single_tile_size,
        in2_t * scaler_tile_size,
        in3_t * bfloat16_tile_size,
        im2_t * single_tile_size,
        reciprocal_CB_size_bytes,
        in_rm_size + out_rm_size,
        a.device()->l1_size_per_core());
    // For input_is_row_major we also allow large_tensor_needed (same L1 logic applies).
    // use_row_major_kernel (row-major gamma/beta) still skips large_tensor check as before.
    if (!use_row_major_kernel || input_is_row_major) {
        if ((gamma.has_value() or beta.has_value() or in_data_format == tt::DataFormat::Float32) and !cb_fits_in_L1) {
            // In the case that the required space is larger than what can be handled by the single pass
            large_tensor_needed = true;
            Wt_next_block_up = with_weights_max_size;
        } else if (!cb_fits_in_L1) {
            large_tensor_needed = true;
            Wt_next_block_up = without_weights_max_size;
        }
    }
    if (large_tensor_needed) {
        in0_t = Wt_next_block_up;
        im0_t = Wt_next_block_up;  // buffer for saving xmm
        im3_t = Wt_next_block_up;  // buffer for xmm^2
        in5_t = Wt_next_block_up;  // buffer for gamma
        in6_t = Wt_next_block_up;  // buffer for beta
        if (b) {
            im6_t = Wt_next_block_up;
            in0_t = 2 * block_size;
        }
        if (input_is_row_major) {
            out0_t = Wt_next_block_up;  // keep in sync with capped value
        }
    }

    if (input_is_row_major && large_tensor_needed) {
        // layernorm_large_tensor.cpp interleaves tilize / compute / untilize per block, so
        // cb_in and cb_out only ever hold one block at a time.  Move the double-buffering to
        // cb_in_rm / cb_out_rm so the reader prefetches the next DRAM block while compute
        // processes the current one, and the writer drains concurrently.
        //
        // layernorm.cpp (large_tensor_needed=false) pre-fills ALL Wt tiles into cb_in before
        // starting the variance loop, and similarly accumulates all Wt tiles in cb_out before
        // UNTILIZE_OUT.  Those paths must keep cb_in = cb_out = Wt_next_block_up.
        in0_t = block_size;
        out0_t = block_size;
    }

    // When the input is ROW_MAJOR and float32, the in-flight tilize_block path requires
    // fp32_dest_acc_en=True.  Without it, UNPACK's SRCA register file is 16-bit and
    // silently truncates every float32 value to bfloat16 before it reaches DST, producing
    // wrong tilized tiles and effectively garbage output.
    TT_FATAL(
        !(input_is_row_major && in_data_format == tt::DataFormat::Float32 && !fp32_dest_acc_en),
        "ROW_MAJOR float32 input requires fp32_dest_acc_en=True in the compute kernel config "
        "(SRCA is 16-bit when fp32_dest_acc_en=False, silently truncating float32 to bfloat16 during tilize)");

    TT_FATAL(in0_t % block_size == 0, "Buffer size in0_t ({}) must be divisible by block_size ({})", in0_t, block_size);
    TT_FATAL(in1_t % block_size == 0, "Buffer size in1_t ({}) must be divisible by block_size ({})", in1_t, block_size);
    TT_FATAL(
        out0_t % block_size == 0, "Buffer size out0_t ({}) must be divisible by block_size ({})", out0_t, block_size);
    TT_FATAL(im0_t % block_size == 0, "Buffer size im0_t ({}) must be divisible by block_size ({})", im0_t, block_size);
    TT_FATAL(im3_t % block_size == 0, "Buffer size im3_t ({}) must be divisible by block_size ({})", im3_t, block_size);
    TT_FATAL(in5_t % block_size == 0, "Buffer size in5_t ({}) must be divisible by block_size ({})", in5_t, block_size);
    TT_FATAL(in6_t % block_size == 0, "Buffer size in6_t ({}) must be divisible by block_size ({})", in6_t, block_size);
    TT_FATAL(im6_t % block_size == 0, "Buffer size im6_t ({}) must be divisible by block_size ({})", im6_t, block_size);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto use_welford_and_not_rms_norm = use_welford && !rms_norm;
    const auto fuse_pre_add = b.has_value();

    // ========================================================================
    //  Metal 2.0 ProgramSpec construction
    //
    //  The legacy `cb_named_args` table (one NamedCompileTimeArgs map handed verbatim to all three
    //  KernelDescriptors) dissolves here into per-KernelSpec DFB bindings: each kernel declares only
    //  the DFBs it actually uses, with its PRODUCER/CONSUMER role. Conditionally-used DFBs are bound
    //  only when their condition holds; the kernels #ifdef-gate the matching dfb::<name> reference
    //  (Conditional / optional DFB bindings pattern). Buffer-address RTAs and the TensorAccessorArgs
    //  CTA plumbing are replaced by TensorParameter / TensorBinding.
    // ========================================================================
    const bool do_gamma = gamma.has_value();
    const bool do_beta = beta.has_value();

    // Welford-fp32 aliasing classification (unchanged from legacy; see the long-form rationale
    // preserved in the kernels). cb_x keeps the default unpack format so post-welford FPU binary
    // ops read via SrcA; the welford-alias index gets UnpackToDestFp32. Deliberately disabled for
    // fused-pre-add + large_tensor (cb_x there already lost precision through the FPU add).
    const bool welford_fp32_alias = use_welford_and_not_rms_norm && in_data_format == tt::DataFormat::Float32 &&
                                    !(fuse_pre_add && large_tensor_needed);
    // Separate alias on cb_ex (c_18) / cb_ex2 (c_19) for the mean / M2 sliding-window accumulators
    // spilled between blocks in layernorm_large_tensor_welford.cpp::welford_fuse_pre_add.
    const bool welford_state_fp32_alias = use_welford_and_not_rms_norm && fuse_pre_add && large_tensor_needed &&
                                          cb_data_format == tt::DataFormat::Float32;

    const bool float32_reduction = fp32_dest_acc_en && !legacy_reduction;

    // ---- Kernel source selection (runtime-selected per attributes; logic unchanged) ----
    const std::string kDataflowDir = "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/";
    const std::string kComputeDir = "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/";

    std::filesystem::path reader_kernel_path;
    if (large_tensor_needed) {
        reader_kernel_path = use_welford_and_not_rms_norm
                                 ? kDataflowDir + "reader_unary_interleaved_ln_large_tensor_welford.cpp"
                                 : kDataflowDir + "reader_unary_interleaved_ln_large_tensor.cpp";
    } else {
        reader_kernel_path = use_row_major_kernel ? kDataflowDir + "reader_unary_interleaved_ln_rm_gb.cpp"
                                                  : kDataflowDir + "reader_unary_interleaved_ln.cpp";
    }

    std::filesystem::path writer_kernel_path =
        input_is_row_major ? kDataflowDir + "writer_unary_interleaved_start_id_blocked_rm_output.cpp"
                           : kDataflowDir + "writer_unary_interleaved_start_id_blocked.cpp";

    std::filesystem::path compute_kernel_path =
        (large_tensor_needed && (!use_row_major_kernel || input_is_row_major))
            ? (use_welford_and_not_rms_norm ? kComputeDir + "layernorm_large_tensor_welford.cpp"
                                            : kComputeDir + "layernorm_large_tensor.cpp")
            : (use_welford_and_not_rms_norm ? kComputeDir + "layernorm_welford.cpp" : kComputeDir + "layernorm.cpp");

    // ---- Defines (FUSE_*/RMSNORM/TILIZE_IN/UNTILIZE_OUT/ACTIVATION + conditional-binding gates) ----
    using Defines = KernelSpec::CompilerOptions::Defines;
    Defines reader_defines;
    Defines compute_defines;
    if (fuse_pre_add) {
        reader_defines.insert({"FUSE_PRE_ADD", "1"});
        // The welford compute also gets FUSE_PRE_ADD as a define (legacy only set it for the
        // non-welford compute): the conditional DFBs cb_x / cb_inb / cb_x_welford(fused) must be
        // #ifdef-gated kernel-side, which requires a preprocessor flag, not just the CTA.
        compute_defines.insert({"FUSE_PRE_ADD", "1"});
    }
    if (do_gamma) {
        reader_defines.insert({"FUSE_GAMMA", "1"});
        // The compute kernel also needs FUSE_GAMMA/FUSE_BETA as preprocessor gates (legacy used the
        // do_gamma/do_beta CTAs only): cb_gamma/cb_beta/cb_fusion are conditionally bound DFBs, so
        // their dfb::<name> references must be #ifdef-gated, not just if-constexpr'd.
        compute_defines.insert({"FUSE_GAMMA", "1"});
    }
    if (do_beta) {
        reader_defines.insert({"FUSE_BETA", "1"});
        compute_defines.insert({"FUSE_BETA", "1"});
    }
    if (rms_norm) {
        reader_defines.insert({"RMSNORM", "1"});
        compute_defines.insert({"RMSNORM", "1"});
    }
    if (input_is_row_major) {
        reader_defines.insert({"TILIZE_IN", "1"});
        compute_defines.insert({"TILIZE_IN", "1"});
        compute_defines.insert({"UNTILIZE_OUT", "1"});
    }
    if (operation_attributes.fused_activation.has_value()) {
        const auto& act = operation_attributes.fused_activation.value();
        auto act_defines =
            ttnn::operations::unary::utils::get_defines(act.op_type, act.params, "ACTIVATION", "i", output.dtype());
        for (auto& [key, val] : act_defines) {
            compute_defines.insert({key, val});
        }
    }
    // Conditional-binding preprocessor gates for the welford-fp32 aliased DFBs. The kernels
    // #ifdef-gate the dfb::cb_x_welford / cb_ex_welford / cb_ex2_welford references, which only
    // exist when the host binds them (below).
    if (welford_fp32_alias) {
        reader_defines.insert({"WELFORD_FP32_ALIAS", "1"});
        compute_defines.insert({"WELFORD_FP32_ALIAS", "1"});
    }
    if (welford_state_fp32_alias) {
        compute_defines.insert({"WELFORD_STATE_FP32_ALIAS", "1"});
    }

    // ---- DataflowBufferSpecs (one per active legacy CB; conditional construction mirrors legacy) ----
    Group<DataflowBufferSpec> dfbs;
    auto add_dfb = [&](const DFBSpecName& id,
                       uint32_t entry_size,
                       uint32_t num_entries,
                       tt::DataFormat fmt,
                       Group<DFBSpecName> alias_with = {},
                       std::optional<TensorParamName> borrowed = std::nullopt) {
        DataflowBufferSpec spec{
            .unique_id = id,
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = fmt,
        };
        spec.borrowed_from = borrowed;
        spec.advanced_options.alias_with = std::move(alias_with);
        dfbs.push_back(std::move(spec));
    };

    // c_0 cb_in (+ c_29 alias when non-fused welford-fp32)
    add_dfb(
        DFB_IN,
        in_single_tile_size,
        in0_t,
        in_data_format,
        (welford_fp32_alias && !fuse_pre_add) ? Group<DFBSpecName>{DFB_X_WELFORD} : Group<DFBSpecName>{});
    // c_16 cb_out
    add_dfb(DFB_OUT, out_single_tile_size, out0_t, out_data_format);
    // c_18 cb_ex (+ c_30 alias) — if !rms_norm
    if (!rms_norm) {
        add_dfb(
            DFB_EX,
            single_tile_size,
            im1_t,
            cb_data_format,
            welford_state_fp32_alias ? Group<DFBSpecName>{DFB_EX_WELFORD} : Group<DFBSpecName>{});
    }
    // c_2 cb_scaler — if !use_welford
    if (!use_welford) {
        add_dfb(DFB_SCALER, scaler_tile_size, in2_t, scaler_cb_data_format);
    }
    // c_3 cb_eps
    add_dfb(DFB_EPS, bfloat16_tile_size, in3_t, tt::DataFormat::Float16_b);
    // c_19 cb_ex2 (+ c_31 alias) — always
    add_dfb(
        DFB_EX2,
        single_tile_size,
        im2_t,
        cb_data_format,
        welford_state_fp32_alias ? Group<DFBSpecName>{DFB_EX2_WELFORD} : Group<DFBSpecName>{});
    // c_24 cb_xmm — if !rms_norm || fuse || large
    if (!rms_norm || fuse_pre_add || large_tensor_needed) {
        add_dfb(DFB_XMM, single_tile_size, im0_t, cb_data_format);
    }
    // c_20 cb_xmm2 — if !use_welford
    if (!use_welford) {
        add_dfb(DFB_XMM2, single_tile_size, im3_t, cb_data_format);
    }
    // c_21 cb_ex2pe — always
    add_dfb(DFB_EX2PE, single_tile_size, im4_t, cb_data_format);
    // c_26 cb_accumulate — if large && !welford
    if (large_tensor_needed && !use_welford) {
        const auto acc_fmt = float32_reduction ? tt::DataFormat::Float32 : cb_data_format;
        add_dfb(DFB_ACCUMULATE, tt::tile_size(acc_fmt), 1, acc_fmt);
    }
    // c_27 cb_in_rm — if input is ROW_MAJOR
    if (input_is_row_major) {
        add_dfb(DFB_IN_RM, in_single_tile_size, in_rm_tiles, in_data_format);
    }
    // c_28 cb_out_rm — if input is ROW_MAJOR
    if (input_is_row_major) {
        add_dfb(DFB_OUT_RM, out_single_tile_size, out_rm_tiles, out_data_format);
    }
    // c_22 cb_fusion — if gamma || beta
    if (do_gamma || do_beta) {
        add_dfb(DFB_FUSION, single_tile_size, im5_t, cb_data_format);
    }
    // c_5 cb_gamma — if gamma
    if (do_gamma) {
        add_dfb(DFB_GAMMA, gamma_single_tile_size, in5_t, gamma_cb_data_format);
    }
    // c_6 cb_beta — if beta
    if (do_beta) {
        add_dfb(DFB_BETA, beta_single_tile_size, in6_t, beta_cb_data_format);
    }
    // c_23 cb_x (+ c_29 alias when fused welford-fp32) — if b && !rms_norm
    if (fuse_pre_add && !rms_norm) {
        add_dfb(
            DFB_X,
            single_tile_size,
            im6_t,
            cb_data_format,
            welford_fp32_alias ? Group<DFBSpecName>{DFB_X_WELFORD} : Group<DFBSpecName>{});
    }
    // c_1 cb_inb — if b
    if (fuse_pre_add) {
        add_dfb(DFB_INB, inb_single_tile_size, in1_t, inb_data_format);
    }
    // c_29 cb_x_welford — aliased DFB (only when welford_fp32_alias). Mirrors the primary's total
    // size: cb_x (fused) or cb_in (non-fused). Configured UnpackToDestFp32 below.
    if (welford_fp32_alias) {
        if (fuse_pre_add) {
            add_dfb(DFB_X_WELFORD, single_tile_size, im6_t, cb_data_format, Group<DFBSpecName>{DFB_X});
        } else {
            add_dfb(DFB_X_WELFORD, in_single_tile_size, in0_t, in_data_format, Group<DFBSpecName>{DFB_IN});
        }
    }
    // c_30 / c_31 cb_ex_welford / cb_ex2_welford — aliased DFBs (only when welford_state_fp32_alias)
    if (welford_state_fp32_alias) {
        add_dfb(DFB_EX_WELFORD, single_tile_size, im1_t, cb_data_format, Group<DFBSpecName>{DFB_EX});
        add_dfb(DFB_EX2_WELFORD, single_tile_size, im2_t, cb_data_format, Group<DFBSpecName>{DFB_EX2});
    }
    // c_25 cb_reciprocals — borrowed from the caller-passed reciprocal LUT tensor (welford only)
    if (use_welford) {
        add_dfb(DFB_RECIPROCALS, reciprocal_CB_size_bytes, 1, reciprocal_cb_data_format, {}, PARAM_RECIP);
    }

    auto bind_self_loop = [](Group<DFBBinding>& g, const DFBSpecName& id, const std::string& name) {
        g.push_back(ProducerOf(id, name));
        g.push_back(ConsumerOf(id, name));
    };

    // ---- Reader KernelSpec ----
    Group<DFBBinding> reader_dfb;
    // TILE input: the reader fills cb_in directly. ROW_MAJOR input: the reader instead fills
    // cb_in_rm (below) and the compute kernel tilizes cb_in_rm -> cb_in, so cb_in is a compute
    // self-loop in that path (see compute bindings).
    if (!input_is_row_major) {
        reader_dfb.push_back(ProducerOf(DFB_IN, "cb_in"));
    }
    reader_dfb.push_back(ProducerOf(DFB_EPS, "cb_eps"));
    if (!use_welford) {
        reader_dfb.push_back(ProducerOf(DFB_SCALER, "cb_scaler"));
    }
    if (fuse_pre_add) {
        reader_dfb.push_back(ProducerOf(DFB_INB, "cb_inb"));
    }
    if (do_gamma) {
        reader_dfb.push_back(ProducerOf(DFB_GAMMA, "cb_gamma"));
    }
    if (do_beta) {
        reader_dfb.push_back(ProducerOf(DFB_BETA, "cb_beta"));
    }
    if (input_is_row_major) {
        reader_dfb.push_back(ProducerOf(DFB_IN_RM, "cb_in_rm"));
    }
    // Non-fused TILE welford-fp32: the reader produces the alias (it pushes cb_x_welford alongside
    // cb_in). In the fused case (or ROW_MAJOR, where compute tilizes cb_in) the alias is produced by
    // compute, not the reader.
    if (welford_fp32_alias && !fuse_pre_add && !input_is_row_major) {
        reader_dfb.push_back(ProducerOf(DFB_X_WELFORD, "cb_x_welford"));
    }

    Group<TensorBinding> reader_tensors;
    reader_tensors.push_back({.tensor_parameter_name = PARAM_INPUT, .accessor_name = "input"});
    if (fuse_pre_add) {
        reader_tensors.push_back({.tensor_parameter_name = PARAM_RESIDUAL, .accessor_name = "residual"});
    }
    if (do_gamma) {
        reader_tensors.push_back({.tensor_parameter_name = PARAM_GAMMA, .accessor_name = "gamma"});
    }
    if (do_beta) {
        reader_tensors.push_back({.tensor_parameter_name = PARAM_BETA, .accessor_name = "beta"});
    }

    KernelSpec::CompileTimeArgs reader_cta;
    reader_cta.insert({"block_size", block_size});
    if (!large_tensor_needed) {
        reader_cta.insert({"use_welford", static_cast<uint32_t>(use_welford)});
    }
    reader_cta.insert({"W", W});
    // Trailing per-path scalar (legacy reader_compile_time_args tail). The legacy else-branch
    // pushed tile_size(a), which the TILE-path reader never read (it uses get_tile_size(cb_in));
    // dropped here as dead plumbing.
    if (input_is_row_major) {
        reader_cta.insert({"elem_size_bytes", static_cast<uint32_t>(a.element_size())});
    } else if (do_gamma && gamma.value().layout() == Layout::ROW_MAJOR) {
        reader_cta.insert(
            {"stick_size", static_cast<uint32_t>(gamma.value().padded_shape()[-1] * gamma.value().element_size())});
    } else if (do_beta && beta.value().layout() == Layout::ROW_MAJOR) {
        reader_cta.insert(
            {"stick_size", static_cast<uint32_t>(beta.value().padded_shape()[-1] * beta.value().element_size())});
    }

    KernelSpec::RuntimeArgSchema reader_rta_schema;
    reader_rta_schema.runtime_arg_names = {"NCHt", "Wt", "start_tile_row", "eps"};
    if (input_is_row_major) {
        reader_rta_schema.runtime_arg_names.push_back("H_logical");
    }

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = reader_kernel_path,
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = std::move(reader_dfb),
        .tensor_bindings = std::move(reader_tensors),
        .compile_time_args = std::move(reader_cta),
        .runtime_arg_schema = std::move(reader_rta_schema),
        .hw_config = DataMovementHardwareConfig{.role = DataMovementHardwareConfig::RoleHint::READER},
    };

    // ---- Writer KernelSpec ----
    Group<DFBBinding> writer_dfb;
    if (input_is_row_major) {
        writer_dfb.push_back(ConsumerOf(DFB_OUT_RM, "cb_out_rm"));
    } else {
        writer_dfb.push_back(ConsumerOf(DFB_OUT, "cb_out"));
    }

    KernelSpec::CompileTimeArgs writer_cta;
    writer_cta.insert({"block_size", block_size});
    if (input_is_row_major) {
        writer_cta.insert({"elem_size_bytes", static_cast<uint32_t>(output.element_size())});
    }

    KernelSpec::RuntimeArgSchema writer_rta_schema;
    writer_rta_schema.runtime_arg_names = {"Wt", "num_tile_rows", "start_tile_row"};
    if (input_is_row_major) {
        writer_rta_schema.runtime_arg_names.push_back("H_logical");
    }

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = writer_kernel_path,
        .dfb_bindings = std::move(writer_dfb),
        .tensor_bindings = {{.tensor_parameter_name = PARAM_OUTPUT, .accessor_name = "output"}},
        .compile_time_args = std::move(writer_cta),
        .runtime_arg_schema = std::move(writer_rta_schema),
        .hw_config = DataMovementHardwareConfig{.role = DataMovementHardwareConfig::RoleHint::WRITER},
    };

    // ---- Compute KernelSpec ----
    // Reader→compute inputs are consumed; the normalized output is produced; every intermediate CB
    // is a compute-internal self-loop (produced and consumed within the compute kernel).
    Group<DFBBinding> compute_dfb;
    compute_dfb.push_back(ConsumerOf(DFB_IN, "cb_in"));
    compute_dfb.push_back(ConsumerOf(DFB_EPS, "cb_eps"));
    if (!use_welford) {
        compute_dfb.push_back(ConsumerOf(DFB_SCALER, "cb_scaler"));
    }
    if (fuse_pre_add) {
        compute_dfb.push_back(ConsumerOf(DFB_INB, "cb_inb"));
    }
    if (do_gamma) {
        compute_dfb.push_back(ConsumerOf(DFB_GAMMA, "cb_gamma"));
    }
    if (do_beta) {
        compute_dfb.push_back(ConsumerOf(DFB_BETA, "cb_beta"));
    }
    compute_dfb.push_back(ProducerOf(DFB_OUT, "cb_out"));
    if (input_is_row_major) {
        // ROW_MAJOR: compute tilizes cb_in_rm (produced by the reader) into cb_in — so it *produces*
        // cb_in here (the reader does not, in this path) — and untilizes cb_out into cb_out_rm (so it
        // self-consumes cb_out).
        compute_dfb.push_back(ProducerOf(DFB_IN, "cb_in"));
        compute_dfb.push_back(ConsumerOf(DFB_OUT, "cb_out"));
        compute_dfb.push_back(ProducerOf(DFB_OUT_RM, "cb_out_rm"));
        compute_dfb.push_back(ConsumerOf(DFB_IN_RM, "cb_in_rm"));
    }
    if (!rms_norm) {
        bind_self_loop(compute_dfb, DFB_EX, "cb_ex");
    }
    bind_self_loop(compute_dfb, DFB_EX2, "cb_ex2");
    if (!use_welford) {
        bind_self_loop(compute_dfb, DFB_XMM2, "cb_xmm2");
    }
    bind_self_loop(compute_dfb, DFB_EX2PE, "cb_ex2pe");
    if (!rms_norm || fuse_pre_add || large_tensor_needed) {
        bind_self_loop(compute_dfb, DFB_XMM, "cb_xmm");
    }
    if (do_gamma || do_beta) {
        bind_self_loop(compute_dfb, DFB_FUSION, "cb_fusion");
    }
    if (fuse_pre_add && !rms_norm) {
        bind_self_loop(compute_dfb, DFB_X, "cb_x");
    }
    if (large_tensor_needed && !use_welford) {
        bind_self_loop(compute_dfb, DFB_ACCUMULATE, "cb_accumulate");
    }
    if (welford_fp32_alias) {
        if (fuse_pre_add || input_is_row_major) {
            // Fused: compute produces the post-add result into the alias and consumes it for welford.
            // ROW_MAJOR non-fused: compute produces cb_in (tilize) and thus the alias too, then
            // consumes it — a self-loop in both cases. (TILE non-fused: the reader produces it.)
            bind_self_loop(compute_dfb, DFB_X_WELFORD, "cb_x_welford");
        } else {
            compute_dfb.push_back(ConsumerOf(DFB_X_WELFORD, "cb_x_welford"));
        }
    }
    if (welford_state_fp32_alias) {
        bind_self_loop(compute_dfb, DFB_EX_WELFORD, "cb_ex_welford");
        bind_self_loop(compute_dfb, DFB_EX2_WELFORD, "cb_ex2_welford");
    }
    if (use_welford) {
        compute_dfb.push_back(ConsumerOf(DFB_RECIPROCALS, "cb_reciprocals"));
    }

    // ---- Per-DFB unpack_to_dest_mode ----
    // Under fp32_dest_acc_en, Metal 2.0 requires an explicit unpack mode for EVERY Float32 DFB this
    // compute kernel consumes (legacy defaulted the whole NUM_CIRCULAR_BUFFERS array to Default;
    // the Metal 2.0 map must name each one). Derive Default for each consumed Float32 DFB, then
    // override the precision-preserving DFBs to UnpackToDestFp32.
    ComputeHardwareConfig::UnpackToDestModes unpack_modes;
    if (fp32_dest_acc_en) {
        for (const auto& binding : compute_dfb) {
            if (binding.endpoint_type != DFBEndpointType::CONSUMER) {
                continue;
            }
            for (const auto& d : dfbs) {
                if (d.unique_id == binding.dfb_spec_name && d.data_format_metadata == tt::DataFormat::Float32) {
                    unpack_modes[d.unique_id] = tt::tt_metal::UnpackToDestMode::Default;
                }
            }
        }
    }
    // Precision-preserving overrides (the large-tensor reduce accumulator and the welford fp32 aliases).
    if (float32_reduction && large_tensor_needed && !use_welford) {
        unpack_modes[DFB_ACCUMULATE] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    if (welford_state_fp32_alias) {
        unpack_modes[DFB_EX_WELFORD] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        unpack_modes[DFB_EX2_WELFORD] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    if (welford_fp32_alias) {
        unpack_modes[DFB_X_WELFORD] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    KernelSpec::CompileTimeArgs compute_cta;
    compute_cta.insert({"Wt", Wt});
    compute_cta.insert({"block_size", block_size});
    compute_cta.insert({"do_gamma", static_cast<uint32_t>(do_gamma)});
    compute_cta.insert({"do_beta", static_cast<uint32_t>(do_beta)});
    compute_cta.insert({"fp32_dest_acc_en", static_cast<uint32_t>(fp32_dest_acc_en)});
    if (use_welford_and_not_rms_norm) {
        compute_cta.insert({"W", W});
        compute_cta.insert({"tile_size", static_cast<uint32_t>(ttnn::types::TILE_SIZE)});
        compute_cta.insert({"rms_norm", static_cast<uint32_t>(rms_norm)});
        compute_cta.insert({"fuse_pre_add", static_cast<uint32_t>(fuse_pre_add)});
    } else {
        compute_cta.insert({"float32_reduction", static_cast<uint32_t>(float32_reduction)});
        compute_cta.insert({"legacy_rsqrt", static_cast<uint32_t>(legacy_rsqrt)});
        compute_cta.insert({"W", W});
        compute_cta.insert({"tile_width", tile_width});
    }

    ComputeHardwareConfig compute_hw{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
        .unpack_to_dest_mode = std::move(unpack_modes),
    };

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = compute_kernel_path,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings = std::move(compute_dfb),
        .compile_time_args = std::move(compute_cta),
        .runtime_arg_schema = {.runtime_arg_names = {"NCHt"}},
        .hw_config = compute_hw,
    };

    // ---- Tensor parameters (one per distinct originating tensor; conditionally present) ----
    Group<TensorParameter> tensor_params;
    tensor_params.push_back({.unique_id = PARAM_INPUT, .spec = a.mesh_tensor().tensor_spec()});
    tensor_params.push_back({.unique_id = PARAM_OUTPUT, .spec = output.mesh_tensor().tensor_spec()});
    if (fuse_pre_add) {
        tensor_params.push_back({.unique_id = PARAM_RESIDUAL, .spec = b.value().mesh_tensor().tensor_spec()});
    }
    if (do_gamma) {
        tensor_params.push_back({.unique_id = PARAM_GAMMA, .spec = gamma.value().mesh_tensor().tensor_spec()});
    }
    if (do_beta) {
        tensor_params.push_back({.unique_id = PARAM_BETA, .spec = beta.value().mesh_tensor().tensor_spec()});
    }
    if (use_welford) {
        tensor_params.push_back({.unique_id = PARAM_RECIP, .spec = recip_tensor.value().mesh_tensor().tensor_spec()});
    }

    // ---- Per-node runtime arg values (paired with the schemas above) ----
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    KernelRunArgs compute_run{.kernel = COMPUTE_KERNEL};

    const uint32_t eps_bits = std::bit_cast<uint32_t>(eps);
    uint32_t curr_row = 0;
    auto all_core_coords = corerange_to_cores(all_cores, num_cores, true);
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = all_core_coords[i];

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        const uint32_t tile_offset = curr_row * Wt;
        // Merged readers (rm_and_tile, large_tensor_rm_and_tile) use a unified start_tile_row = curr_row.
        // Legacy kernels (welford large-tensor, rm_gb) still expect tile_offset = curr_row * Wt.
        const bool using_legacy_tile_reader =
            (use_welford_and_not_rms_norm && large_tensor_needed) || (use_row_major_kernel && !input_is_row_major);
        const uint32_t reader_start = using_legacy_tile_reader ? tile_offset : curr_row;
        // For the RM output writer start_tile_row is the tile-row index; the tile writer wants the
        // flat tile offset.
        const uint32_t writer_start = input_is_row_major ? curr_row : tile_offset;

        KernelRunArgs::RuntimeArgValues reader_args{
            {"NCHt", num_tile_rows_per_core}, {"Wt", Wt}, {"start_tile_row", reader_start}, {"eps", eps_bits}};
        if (input_is_row_major) {
            reader_args.insert({"H_logical", H_logical});
        }
        reader_run.runtime_arg_values.push_back({core, reader_args});

        KernelRunArgs::RuntimeArgValues writer_args{
            {"Wt", Wt}, {"num_tile_rows", num_tile_rows_per_core}, {"start_tile_row", writer_start}};
        if (input_is_row_major) {
            writer_args.insert({"H_logical", H_logical});
        }
        writer_run.runtime_arg_values.push_back({core, writer_args});

        compute_run.runtime_arg_values.push_back({core, {{"NCHt", num_tile_rows_per_core}}});

        curr_row += num_tile_rows_per_core;
    }

    // ---- Assemble the ProgramSpec + ProgramRunArgs ----
    WorkUnitSpec work_unit{
        .name = "layernorm_interleaved",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = all_cores,
    };

    ProgramSpec spec{
        .name = "layernorm_interleaved",
        .kernels = {std::move(reader_spec), std::move(writer_spec), std::move(compute_spec)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_params),
        .work_units = {std::move(work_unit)},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};
    run_args.tensor_args.insert({PARAM_INPUT, TensorArgument{std::cref(a.mesh_tensor())}});
    run_args.tensor_args.insert({PARAM_OUTPUT, TensorArgument{std::cref(output.mesh_tensor())}});
    if (fuse_pre_add) {
        run_args.tensor_args.insert({PARAM_RESIDUAL, TensorArgument{std::cref(b.value().mesh_tensor())}});
    }
    if (do_gamma) {
        run_args.tensor_args.insert({PARAM_GAMMA, TensorArgument{std::cref(gamma.value().mesh_tensor())}});
    }
    if (do_beta) {
        run_args.tensor_args.insert({PARAM_BETA, TensorArgument{std::cref(beta.value().mesh_tensor())}});
    }
    if (use_welford) {
        run_args.tensor_args.insert({PARAM_RECIP, TensorArgument{std::cref(recip_tensor.value().mesh_tensor())}});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

CoreRangeSet LayerNormMultiCoreProgramFactory::default_core_range(IDevice* device) {
    auto grid_size = device->compute_with_storage_grid_size();
    return CoreRangeSet({CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1})});
}

}  // namespace ttnn::prim
