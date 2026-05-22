// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>

#include <optional>
#include <bit>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental::metal2_host_api;

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

// Resource identifier constants (kernel / DFB / tensor / work-unit names).
constexpr const char* READER_KERNEL = "reader";
constexpr const char* WRITER_KERNEL = "writer";
constexpr const char* COMPUTE_KERNEL = "compute";
constexpr const char* MAIN_WORK_UNIT = "main";

// DFB unique-ids (match the legacy CBIndex assignments by symbol; the numeric index
// is gone in Metal 2.0)
constexpr const char* DFB_CB_IN = "cb_in";
constexpr const char* DFB_CB_INB = "cb_inb";
constexpr const char* DFB_CB_SCALER = "cb_scaler";
constexpr const char* DFB_CB_EPS = "cb_eps";
constexpr const char* DFB_CB_GAMMA = "cb_gamma";
constexpr const char* DFB_CB_BETA = "cb_beta";
constexpr const char* DFB_CB_OUT = "cb_out";
constexpr const char* DFB_CB_EX = "cb_ex";                    // CB 18: E[x]
constexpr const char* DFB_CB_EX2 = "cb_ex2";                  // CB 19: E[(x-E[x])^2]
constexpr const char* DFB_CB_XMM2 = "cb_xmm2";                // CB 20: xmm^2
constexpr const char* DFB_CB_EX2PE = "cb_ex2pe";              // CB 21: E[(x-E[x])^2]+eps
constexpr const char* DFB_CB_FUSION = "cb_fusion";            // CB 22
constexpr const char* DFB_CB_X = "cb_x";                      // CB 23: x = a + b
constexpr const char* DFB_CB_XMM = "cb_xmm";                  // CB 24: x - E[x]
constexpr const char* DFB_CB_RECIPROCALS = "cb_reciprocals";  // CB 25: Welford recip LUT
constexpr const char* DFB_CB_ACCUMULATE = "cb_accumulate";    // CB 26: large-tensor acc
constexpr const char* DFB_CB_IN_RM = "cb_in_rm";              // CB 27: RM staging in
constexpr const char* DFB_CB_OUT_RM = "cb_out_rm";            // CB 28: RM staging out

// TensorParameter unique-ids
constexpr const char* TP_INPUT_A = "input_a";
constexpr const char* TP_RESIDUAL_B = "residual_b";
constexpr const char* TP_GAMMA = "gamma";
constexpr const char* TP_BETA = "beta";
constexpr const char* TP_OUTPUT = "output";
constexpr const char* TP_RECIP = "recip";

// Helper: build a DFB consumer binding.
m2::KernelSpec::DFBBinding ConsumerDFB(const char* dfb_name, const char* accessor_name) {
    return m2::KernelSpec::DFBBinding{
        .dfb_spec_name = dfb_name,
        .local_accessor_name = accessor_name,
        .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER};
}

m2::KernelSpec::DFBBinding ProducerDFB(const char* dfb_name, const char* accessor_name) {
    return m2::KernelSpec::DFBBinding{
        .dfb_spec_name = dfb_name,
        .local_accessor_name = accessor_name,
        .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER};
}

m2::DataflowBufferSpec MakeDFB(
    const char* unique_id,
    uint32_t entry_size,
    uint32_t num_entries,
    tt::DataFormat data_format,
    std::optional<m2::TensorParameterName> borrowed_from = std::nullopt) {
    m2::DataflowBufferSpec dfb{
        .unique_id = unique_id,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = data_format};
    if (borrowed_from.has_value()) {
        dfb.borrowed_from = *borrowed_from;
    }
    // Implicit sync is a Gen2-only feature; this op targets WH/BH (Gen1).
    dfb.disable_implicit_sync = true;
    return dfb;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts LayerNormMultiCoreProgramFactory::create_program_spec(
    const LayerNormParams& operation_attributes, const LayerNormInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

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
    if (input_is_row_major) {
        Hp = tt::round_up(Hp, TILE_HEIGHT);
    }
    const uint32_t H_logical = static_cast<uint32_t>(NC) * static_cast<uint32_t>(logical_shape[-2]);

    IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    uint32_t Wt = Wp / tile_width;
    uint32_t Ht = Hp / tile_height;

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

    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;
    if (b) {
        inb_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.value().dtype());
        inb_single_tile_size = tt::tile_size(inb_data_format);
    }

    uint32_t num_tile_rows = NC * Ht;

    CoreRangeSet requested_cores = default_core_range(device);

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = split_work_to_cores(requested_cores, num_tile_rows, true /* row_wise */);

    uint32_t reciprocal_CB_size_bytes = 0;
    if (use_welford) {
        TT_FATAL(tensor_args.recip_tensor.has_value(), "Reciprocal tensor not provided for Welford layernorm");
        reciprocal_CB_size_bytes = tensor_args.recip_tensor->buffer()->aligned_size_per_bank();
    }

    auto use_row_major_kernel = (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) or
                                (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR);
    uint32_t Wt_next_block_up = tt::round_up(Wt, block_size);
    uint32_t in0_t = Wt_next_block_up;
    uint32_t in1_t = block_size * 2;
    uint32_t out0_t = input_is_row_major ? Wt_next_block_up : block_size * 2;
    uint32_t im0_t = Wt_next_block_up;
    uint32_t im3_t = Wt_next_block_up;
    uint32_t in5_t = Wt_next_block_up;
    uint32_t in6_t = Wt_next_block_up;
    uint32_t im6_t = block_size * 2;
    if (b) {
        im6_t = Wt_next_block_up;
        in0_t = 2 * block_size;
    }
    uint32_t im5_t = 2 * block_size;
    uint32_t im4_t = 8;
    uint32_t im1_t = 2;
    uint32_t in2_t = 2;
    uint32_t in3_t = 2;
    uint32_t im2_t = 2;

    bool large_tensor_needed = false;
    const uint32_t with_weights_max_size = 56 * bfloat16_tile_size / single_tile_size;
    const uint32_t without_weights_max_size = 112 * bfloat16_tile_size / single_tile_size;
    const uint32_t in_rm_tiles = input_is_row_major ? 2 * block_size : 0;
    const uint32_t in_rm_size = in_rm_tiles * in_single_tile_size;
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
    if (!use_row_major_kernel || input_is_row_major) {
        if ((gamma.has_value() or beta.has_value() or in_data_format == tt::DataFormat::Float32) and !cb_fits_in_L1) {
            large_tensor_needed = true;
            Wt_next_block_up = with_weights_max_size;
        } else if (!cb_fits_in_L1) {
            large_tensor_needed = true;
            Wt_next_block_up = without_weights_max_size;
        }
    }
    if (large_tensor_needed) {
        in0_t = Wt_next_block_up;
        im0_t = Wt_next_block_up;
        im3_t = Wt_next_block_up;
        in5_t = Wt_next_block_up;
        in6_t = Wt_next_block_up;
        if (b) {
            im6_t = Wt_next_block_up;
            in0_t = 2 * block_size;
        }
        if (input_is_row_major) {
            out0_t = Wt_next_block_up;
        }
    }

    if (input_is_row_major && large_tensor_needed) {
        in0_t = block_size;
        out0_t = block_size;
    }

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

    const auto use_welford_and_not_rms_norm = use_welford && !rms_norm;
    const auto fuse_pre_add = b.has_value();

    ////////////////////////////////////////////////////////////////////////////
    // Select kernel paths (legacy behavior — runtime dispatch on flags)
    ////////////////////////////////////////////////////////////////////////////
    const char* reader_kernel_path = nullptr;
    if (large_tensor_needed) {
        reader_kernel_path = use_welford_and_not_rms_norm
                                 ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
                                   "reader_unary_interleaved_ln_large_tensor_welford.cpp"
                                 : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
                                   "reader_unary_interleaved_ln_large_tensor.cpp";
    } else {
        reader_kernel_path = use_row_major_kernel
                                 ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
                                   "reader_unary_interleaved_ln_rm_gb.cpp"
                                 : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
                                   "reader_unary_interleaved_ln.cpp";
    }

    const char* writer_kernel_path = input_is_row_major
                                         ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
                                           "writer_unary_interleaved_start_id_blocked_rm_output.cpp"
                                         : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
                                           "writer_unary_interleaved_start_id_blocked.cpp";

    bool float32_reduction = fp32_dest_acc_en && !legacy_reduction;
    const char* compute_kernel_path =
        (large_tensor_needed && (!use_row_major_kernel || input_is_row_major))
            ? (use_welford_and_not_rms_norm ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/"
                                              "layernorm_large_tensor_welford.cpp"
                                            : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/"
                                              "layernorm_large_tensor.cpp")
            : (use_welford_and_not_rms_norm
                   ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_welford.cpp"
                   : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp");

    ////////////////////////////////////////////////////////////////////////////
    // Build DataflowBufferSpecs
    ////////////////////////////////////////////////////////////////////////////
    std::vector<m2::DataflowBufferSpec> dfbs;

    // Tile-count → entry_count / entry_size: in legacy code the CB total_size was
    // num_tiles * tile_size. In Metal 2.0 we declare num_entries = num_tiles and
    // entry_size = tile_size to keep the wire-level format identical.
    dfbs.push_back(MakeDFB(DFB_CB_IN, in_single_tile_size, in0_t, in_data_format));
    dfbs.push_back(MakeDFB(DFB_CB_OUT, out_single_tile_size, out0_t, out_data_format));
    if (!rms_norm) {
        dfbs.push_back(MakeDFB(DFB_CB_EX, single_tile_size, im1_t, cb_data_format));
    }
    if (!use_welford) {
        dfbs.push_back(MakeDFB(DFB_CB_SCALER, scaler_tile_size, in2_t, scaler_cb_data_format));
    }
    dfbs.push_back(MakeDFB(DFB_CB_EPS, bfloat16_tile_size, in3_t, tt::DataFormat::Float16_b));
    dfbs.push_back(MakeDFB(DFB_CB_EX2, single_tile_size, im2_t, cb_data_format));
    if (!rms_norm || fuse_pre_add || large_tensor_needed) {
        dfbs.push_back(MakeDFB(DFB_CB_XMM, single_tile_size, im0_t, cb_data_format));
    }
    if (!use_welford) {
        dfbs.push_back(MakeDFB(DFB_CB_XMM2, single_tile_size, im3_t, cb_data_format));
    }
    dfbs.push_back(MakeDFB(DFB_CB_EX2PE, single_tile_size, im4_t, cb_data_format));
    if (large_tensor_needed && !use_welford) {
        const auto large_tensor_acc_data_format = float32_reduction ? tt::DataFormat::Float32 : cb_data_format;
        dfbs.push_back(
            MakeDFB(DFB_CB_ACCUMULATE, tt::tile_size(large_tensor_acc_data_format), 1, large_tensor_acc_data_format));
    }
    if (input_is_row_major) {
        dfbs.push_back(MakeDFB(DFB_CB_IN_RM, in_single_tile_size, in_rm_tiles, in_data_format));
    }
    if (input_is_row_major) {
        dfbs.push_back(MakeDFB(DFB_CB_OUT_RM, out_single_tile_size, out_rm_tiles, out_data_format));
    }
    if (gamma.has_value() || beta.has_value()) {
        dfbs.push_back(MakeDFB(DFB_CB_FUSION, single_tile_size, im5_t, cb_data_format));
    }
    if (gamma.has_value()) {
        dfbs.push_back(MakeDFB(DFB_CB_GAMMA, gamma_single_tile_size, in5_t, gamma_cb_data_format));
    }
    if (beta.has_value()) {
        dfbs.push_back(MakeDFB(DFB_CB_BETA, beta_single_tile_size, in6_t, beta_cb_data_format));
    }
    if (b) {
        if (!rms_norm) {
            dfbs.push_back(MakeDFB(DFB_CB_X, single_tile_size, im6_t, cb_data_format));
        }
        dfbs.push_back(MakeDFB(DFB_CB_INB, inb_single_tile_size, in1_t, inb_data_format));
    }
    if (use_welford) {
        // Borrowed-memory DFB backing the reciprocal LUT tensor.
        dfbs.push_back(MakeDFB(
            DFB_CB_RECIPROCALS,
            reciprocal_CB_size_bytes,
            1,
            reciprocal_cb_data_format,
            m2::TensorParameterName{TP_RECIP}));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Build TensorParameters
    ////////////////////////////////////////////////////////////////////////////
    std::vector<m2::TensorParameter> tensor_parameters;
    tensor_parameters.push_back({.unique_id = TP_INPUT_A, .spec = a.tensor_spec()});
    tensor_parameters.push_back({.unique_id = TP_OUTPUT, .spec = output.tensor_spec()});
    if (b.has_value()) {
        tensor_parameters.push_back({.unique_id = TP_RESIDUAL_B, .spec = b.value().tensor_spec()});
    }
    if (gamma.has_value()) {
        tensor_parameters.push_back({.unique_id = TP_GAMMA, .spec = gamma.value().tensor_spec()});
    }
    if (beta.has_value()) {
        tensor_parameters.push_back({.unique_id = TP_BETA, .spec = beta.value().tensor_spec()});
    }
    if (use_welford) {
        tensor_parameters.push_back({.unique_id = TP_RECIP, .spec = tensor_args.recip_tensor->tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    // Build KernelSpecs
    ////////////////////////////////////////////////////////////////////////////

    // ---------- Reader ----------
    m2::KernelSpec reader_spec;
    reader_spec.unique_id = READER_KERNEL;
    reader_spec.source = m2::KernelSpec::SourceFilePath{reader_kernel_path};
    reader_spec.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config = m2::DataMovementConfiguration::Gen1DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}};
    // Named CTAs replace the legacy positional list. Slot meaning differs across the
    // four reader variants, so we set `block_size`, `W`, and (variant-specific) others.
    reader_spec.compile_time_arg_bindings = {
        {"block_size", block_size},
        {"W", W},
    };
    if (!large_tensor_needed) {
        // "ln" + "ln_rm_gb" readers take a use_welford CTA in slot[1] of the legacy
        // layout. (large-tensor variant doesn't take it.)
        reader_spec.compile_time_arg_bindings.push_back({"use_welford", static_cast<uint32_t>(use_welford)});
    }
    // Variant-specific last CTA: elem_size_bytes (rm input or large_tensor rm),
    // gamma_stick_size (rm_gb with gamma rm), beta_stick_size (rm_gb with beta rm),
    // or tile_size of input dtype (rm_gb otherwise).
    if (input_is_row_major) {
        reader_spec.compile_time_arg_bindings.push_back({"elem_size_bytes", static_cast<uint32_t>(a.element_size())});
    } else if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        const uint32_t gamma_stick_size = gamma.value().padded_shape()[-1] * gamma.value().element_size();
        reader_spec.compile_time_arg_bindings.push_back({"gamma_or_beta_stick", gamma_stick_size});
    } else if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        const uint32_t beta_stick_size = beta.value().padded_shape()[-1] * beta.value().element_size();
        reader_spec.compile_time_arg_bindings.push_back({"gamma_or_beta_stick", beta_stick_size});
    } else {
        reader_spec.compile_time_arg_bindings.push_back(
            {"input_tile_size", tt::tile_size(datatype_to_dataformat_converter(a.dtype()))});
    }
    // Reader DFB bindings — producer for input/residual/gamma/beta CBs (and scaler/eps).
    reader_spec.dfb_bindings.push_back(ProducerDFB(DFB_CB_IN, "cb_in"));
    if (b.has_value()) {
        reader_spec.dfb_bindings.push_back(ProducerDFB(DFB_CB_INB, "cb_inb"));
    }
    if (gamma.has_value()) {
        reader_spec.dfb_bindings.push_back(ProducerDFB(DFB_CB_GAMMA, "cb_gamma"));
    }
    if (beta.has_value()) {
        reader_spec.dfb_bindings.push_back(ProducerDFB(DFB_CB_BETA, "cb_beta"));
    }
    if (!use_welford) {
        // Scaler is produced by the reader via dataflow_kernel_lib helpers.
        reader_spec.dfb_bindings.push_back(ProducerDFB(DFB_CB_SCALER, "cb_scaler"));
    }
    reader_spec.dfb_bindings.push_back(ProducerDFB(DFB_CB_EPS, "cb_eps"));
    if (input_is_row_major) {
        reader_spec.dfb_bindings.push_back(ProducerDFB(DFB_CB_IN_RM, "cb_in_rm"));
    }
    // Tensor bindings on the reader.
    reader_spec.tensor_bindings.push_back({.tensor_parameter_name = TP_INPUT_A, .accessor_name = "src_a"});
    if (b.has_value()) {
        reader_spec.tensor_bindings.push_back({.tensor_parameter_name = TP_RESIDUAL_B, .accessor_name = "src_b"});
    }
    if (gamma.has_value()) {
        reader_spec.tensor_bindings.push_back({.tensor_parameter_name = TP_GAMMA, .accessor_name = "gamma"});
    }
    if (beta.has_value()) {
        reader_spec.tensor_bindings.push_back({.tensor_parameter_name = TP_BETA, .accessor_name = "beta"});
    }
    if (use_welford) {
        // Ghost TensorBinding to satisfy the spec validator (every TensorParameter
        // requires ≥1 TensorBinding). The reader kernel does not actually access
        // `ta::recip` — the recip tensor's L1 data is consumed via the borrowed-memory
        // `cb_reciprocals` DFB on the compute kernel.
        reader_spec.tensor_bindings.push_back({.tensor_parameter_name = TP_RECIP, .accessor_name = "recip"});
    }
    // Reader RTAs (buffer addresses are gone — auto-injected from TensorBinding).
    reader_spec.runtime_arguments_schema.named_runtime_args = {"NCHt", "Wt", "start_tile_row", "eps"};
    if (input_is_row_major) {
        reader_spec.runtime_arguments_schema.named_runtime_args.push_back("H_logical");
    }
    // Defines (legacy macro guards into kernel sources)
    if (fuse_pre_add) {
        reader_spec.compiler_options.defines.push_back({"FUSE_PRE_ADD", "1"});
    }
    if (gamma.has_value()) {
        reader_spec.compiler_options.defines.push_back({"FUSE_GAMMA", "1"});
    }
    if (beta.has_value()) {
        reader_spec.compiler_options.defines.push_back({"FUSE_BETA", "1"});
    }
    if (rms_norm) {
        reader_spec.compiler_options.defines.push_back({"RMSNORM", "1"});
    }
    if (input_is_row_major) {
        reader_spec.compiler_options.defines.push_back({"TILIZE_IN", "1"});
    }
    if (use_welford) {
        // Preprocessor-level guard: when use_welford is true, the reader doesn't bind
        // cb_scaler, so any `if constexpr (!use_welford)` block that references
        // `dfb::cb_scaler` fails at parse-time name lookup. #ifdef gates the block out.
        reader_spec.compiler_options.defines.push_back({"USE_WELFORD", "1"});
    }

    // ---------- Writer ----------
    m2::KernelSpec writer_spec;
    writer_spec.unique_id = WRITER_KERNEL;
    writer_spec.source = m2::KernelSpec::SourceFilePath{writer_kernel_path};
    writer_spec.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config = m2::DataMovementConfiguration::Gen1DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}};
    writer_spec.compile_time_arg_bindings = {{"block_size", block_size}};
    if (input_is_row_major) {
        writer_spec.compile_time_arg_bindings.push_back(
            {"elem_size_bytes", static_cast<uint32_t>(output.element_size())});
    }
    if (input_is_row_major) {
        writer_spec.dfb_bindings.push_back(ConsumerDFB(DFB_CB_OUT_RM, "cb_out_rm"));
    } else {
        writer_spec.dfb_bindings.push_back(ConsumerDFB(DFB_CB_OUT, "cb_out"));
    }
    writer_spec.tensor_bindings.push_back({.tensor_parameter_name = TP_OUTPUT, .accessor_name = "output"});
    writer_spec.runtime_arguments_schema.named_runtime_args = {"Wt", "num_tile_rows", "writer_start"};
    if (input_is_row_major) {
        writer_spec.runtime_arguments_schema.named_runtime_args.push_back("H_logical");
    }

    // ---------- Compute ----------
    m2::KernelSpec compute_spec;
    compute_spec.unique_id = COMPUTE_KERNEL;
    compute_spec.source = m2::KernelSpec::SourceFilePath{compute_kernel_path};
    m2::ComputeConfiguration compute_config{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};
    // Per Metal 2.0 validator: every compute-consumed FP32 DFB requires an explicit
    // unpack_to_dest_mode entry when fp32_dest_acc_en is true. Add Default for each
    // FP32 DFB compute consumes, then override cb_accumulate to UnpackToDestFp32 in
    // the large-tensor non-Welford path.
    if (fp32_dest_acc_en) {
        auto add_default = [&](const char* dfb_name) {
            compute_config.unpack_to_dest_mode.push_back({dfb_name, tt::tt_metal::UnpackToDestMode::Default});
        };
        // cb_data_format == Float32 when fp32_dest_acc_en, so all intermediate DFBs
        // that use cb_data_format need an entry. Mirror the host-side declaration guards.
        if (!rms_norm || fuse_pre_add || large_tensor_needed) {
            add_default(DFB_CB_XMM);
        }
        if (!rms_norm) {
            add_default(DFB_CB_EX);
        }
        add_default(DFB_CB_EX2);
        if (!use_welford) {
            add_default(DFB_CB_XMM2);
        }
        add_default(DFB_CB_EX2PE);
        if (gamma.has_value() || beta.has_value()) {
            add_default(DFB_CB_FUSION);
        }
        if (b.has_value() && !rms_norm) {
            add_default(DFB_CB_X);
        }
        // Inputs / weights: only need an entry if their dtype happens to be FP32.
        if (in_data_format == tt::DataFormat::Float32) {
            add_default(DFB_CB_IN);
        }
        if (b.has_value() && inb_data_format == tt::DataFormat::Float32) {
            add_default(DFB_CB_INB);
        }
        if (gamma.has_value() && gamma_cb_data_format == tt::DataFormat::Float32) {
            add_default(DFB_CB_GAMMA);
        }
        if (beta.has_value() && beta_cb_data_format == tt::DataFormat::Float32) {
            add_default(DFB_CB_BETA);
        }
        // cb_reciprocals is Float32 always (Welford LUT).
        if (use_welford) {
            add_default(DFB_CB_RECIPROCALS);
        }
    }
    if (float32_reduction && large_tensor_needed && !use_welford) {
        // Large-tensor non-Welford accumulator unpacks Float32 → Dest directly.
        // Only valid when the cb_accumulate DFB is actually declared and bound on compute.
        compute_config.unpack_to_dest_mode.push_back(
            {DFB_CB_ACCUMULATE, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
    }
    compute_spec.config_spec = std::move(compute_config);
    compute_spec.compile_time_arg_bindings = {
        {"Wt", Wt},
        {"block_size", block_size},
        {"do_gamma", static_cast<uint32_t>(gamma.has_value())},
        {"do_beta", static_cast<uint32_t>(beta.has_value())},
        {"FLOAT32_DTYPE", static_cast<uint32_t>(fp32_dest_acc_en)},
    };
    if (use_welford_and_not_rms_norm) {
        compute_spec.compile_time_arg_bindings.push_back({"W", W});
        compute_spec.compile_time_arg_bindings.push_back({"TILE_SIZE", ttnn::types::TILE_SIZE});
        compute_spec.compile_time_arg_bindings.push_back({"rms_norm", static_cast<uint32_t>(rms_norm)});
        compute_spec.compile_time_arg_bindings.push_back({"fuse_pre_add", static_cast<uint32_t>(fuse_pre_add)});
    } else {
        compute_spec.compile_time_arg_bindings.push_back(
            {"FLOAT32_REDUCTION", static_cast<uint32_t>(float32_reduction)});
        compute_spec.compile_time_arg_bindings.push_back({"LEGACY_RSQRT", static_cast<uint32_t>(legacy_rsqrt)});
        compute_spec.compile_time_arg_bindings.push_back({"W", W});
        compute_spec.compile_time_arg_bindings.push_back({"tile_width", tile_width});
    }
    // Compute DFB bindings — the compute kernel consumes inputs, produces output, and
    // produces+consumes all intermediates. Per [Pattern: Conditional / optional DFB
    // bindings] (metal2_port_patterns.md), bind unconditionally on the host whenever the
    // DFB exists; the kernel `if constexpr`-gates its uses.
    auto bind_compute_pair = [&](const char* dfb_name, const char* accessor_name) {
        compute_spec.dfb_bindings.push_back(ProducerDFB(dfb_name, accessor_name));
        compute_spec.dfb_bindings.push_back(ConsumerDFB(dfb_name, accessor_name));
        compute_spec.dfb_compute_self_loop_scopes.push_back(
            {.dfb_spec_name = dfb_name, .scope = m2::KernelSpec::DFBComputeSelfLoopScope::Scope::INTRA});
    };
    // Reader → compute (consumer endpoints on compute).
    compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_CB_IN, "cb_in"));
    if (b.has_value()) {
        compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_CB_INB, "cb_inb"));
    }
    if (gamma.has_value()) {
        compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_CB_GAMMA, "cb_gamma"));
    }
    if (beta.has_value()) {
        compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_CB_BETA, "cb_beta"));
    }
    if (!use_welford) {
        compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_CB_SCALER, "cb_scaler"));
    }
    compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_CB_EPS, "cb_eps"));
    if (input_is_row_major) {
        compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_CB_IN_RM, "cb_in_rm"));
    }
    // Compute → writer (producer endpoint on compute).
    compute_spec.dfb_bindings.push_back(ProducerDFB(DFB_CB_OUT, "cb_out"));
    if (input_is_row_major) {
        compute_spec.dfb_bindings.push_back(ProducerDFB(DFB_CB_OUT_RM, "cb_out_rm"));
    }
    // Compute self-loop DFBs (compute is both producer and consumer).
    if (!rms_norm) {
        bind_compute_pair(DFB_CB_EX, "cb_ex");
    }
    bind_compute_pair(DFB_CB_EX2, "cb_ex2");
    if (!rms_norm || fuse_pre_add || large_tensor_needed) {
        bind_compute_pair(DFB_CB_XMM, "cb_xmm");
    }
    if (!use_welford) {
        bind_compute_pair(DFB_CB_XMM2, "cb_xmm2");
    }
    bind_compute_pair(DFB_CB_EX2PE, "cb_ex2pe");
    if (gamma.has_value() || beta.has_value()) {
        bind_compute_pair(DFB_CB_FUSION, "cb_fusion");
    }
    if (b.has_value() && !rms_norm) {
        bind_compute_pair(DFB_CB_X, "cb_x");
    }
    if (large_tensor_needed && !use_welford) {
        bind_compute_pair(DFB_CB_ACCUMULATE, "cb_accumulate");
    }
    if (use_welford) {
        // Welford reciprocal LUT — read-only by compute (data comes from the borrowed
        // recip tensor; no kernel writes to it). The spec validator requires every DFB
        // to have ≥1 PRODUCER binding, so declare a ghost PRODUCER alongside the real
        // CONSUMER. The compute kernel never calls reserve_back/push_back; the data is
        // read directly via the borrowed memory at the DFB's L1 address.
        compute_spec.dfb_bindings.push_back(ProducerDFB(DFB_CB_RECIPROCALS, "cb_reciprocals"));
        compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_CB_RECIPROCALS, "cb_reciprocals"));
    }
    compute_spec.runtime_arguments_schema.named_runtime_args = {"NCHt"};
    if (fuse_pre_add && !use_welford) {
        compute_spec.compiler_options.defines.push_back({"FUSE_PRE_ADD", "1"});
    }
    if (rms_norm) {
        compute_spec.compiler_options.defines.push_back({"RMSNORM", "1"});
    }
    if (input_is_row_major) {
        compute_spec.compiler_options.defines.push_back({"TILIZE_IN", "1"});
        compute_spec.compiler_options.defines.push_back({"UNTILIZE_OUT", "1"});
    }
    if (operation_attributes.fused_activation.has_value()) {
        const auto& act = operation_attributes.fused_activation.value();
        auto act_defines =
            ttnn::operations::unary::utils::get_defines(act.op_type, act.params, "ACTIVATION", "i", output.dtype());
        for (auto& [key, val] : act_defines) {
            compute_spec.compiler_options.defines.push_back({key, val});
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Build WorkUnitSpec
    ////////////////////////////////////////////////////////////////////////////
    m2::WorkUnitSpec work_unit{
        .unique_id = MAIN_WORK_UNIT,
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = all_cores,  // NodeRangeSet is an alias for CoreRangeSet
    };

    ////////////////////////////////////////////////////////////////////////////
    // Assemble ProgramSpec + ProgramRunParams
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec{
        .program_id = "layernorm_interleaved",
        .kernels = {std::move(reader_spec), std::move(writer_spec), std::move(compute_spec)},
        .dataflow_buffers = std::move(dfbs),
        .semaphores = {},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {std::move(work_unit)},
    };

    m2::ProgramRunParams run_params;

    ////////////////////////////////////////////////////////////////////////////
    // Per-node runtime arguments
    ////////////////////////////////////////////////////////////////////////////
    auto bfloat_one_value = bfloat16(1);
    uint32_t packed_one_value = pack_two_bfloat16_into_uint32({bfloat_one_value, bfloat_one_value});
    (void)packed_one_value;  // legacy slot, value is in-kernel today

    const uint32_t eps_u = std::bit_cast<uint32_t>(eps);

    m2::ProgramRunParams::KernelRunParams reader_run{.kernel_spec_name = READER_KERNEL};
    m2::ProgramRunParams::KernelRunParams writer_run{.kernel_spec_name = WRITER_KERNEL};
    m2::ProgramRunParams::KernelRunParams compute_run{.kernel_spec_name = COMPUTE_KERNEL};

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

        uint32_t tile_offset = curr_row * Wt;
        // Legacy reader variants disagree on whether arg[3] is tile_offset or curr_row.
        const bool using_legacy_tile_reader =
            (use_welford_and_not_rms_norm && large_tensor_needed) || (use_row_major_kernel && !input_is_row_major);
        const uint32_t reader_start = using_legacy_tile_reader ? tile_offset : curr_row;

        m2::NodeCoord node{core.x, core.y};

        std::unordered_map<std::string, uint32_t> reader_args = {
            {"NCHt", num_tile_rows_per_core},
            {"Wt", Wt},
            {"start_tile_row", reader_start},
            {"eps", eps_u},
        };
        if (input_is_row_major) {
            reader_args["H_logical"] = H_logical;
        }
        reader_run.named_runtime_args.push_back({.node = node, .args = std::move(reader_args)});

        const uint32_t writer_start = input_is_row_major ? curr_row : tile_offset;
        std::unordered_map<std::string, uint32_t> writer_args = {
            {"Wt", Wt},
            {"num_tile_rows", num_tile_rows_per_core},
            {"writer_start", writer_start},
        };
        if (input_is_row_major) {
            writer_args["H_logical"] = H_logical;
        }
        writer_run.named_runtime_args.push_back({.node = node, .args = std::move(writer_args)});

        compute_run.named_runtime_args.push_back({.node = node, .args = {{"NCHt", num_tile_rows_per_core}}});

        curr_row += num_tile_rows_per_core;
    }

    run_params.kernel_run_params = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};

    ////////////////////////////////////////////////////////////////////////////
    // TensorArgs (per-execution tensor identities)
    ////////////////////////////////////////////////////////////////////////////
    run_params.tensor_args.push_back({.tensor_parameter_name = TP_INPUT_A, .tensor = a.mesh_tensor()});
    run_params.tensor_args.push_back({.tensor_parameter_name = TP_OUTPUT, .tensor = output.mesh_tensor()});
    if (b.has_value()) {
        run_params.tensor_args.push_back({.tensor_parameter_name = TP_RESIDUAL_B, .tensor = b.value().mesh_tensor()});
    }
    if (gamma.has_value()) {
        run_params.tensor_args.push_back({.tensor_parameter_name = TP_GAMMA, .tensor = gamma.value().mesh_tensor()});
    }
    if (beta.has_value()) {
        run_params.tensor_args.push_back({.tensor_parameter_name = TP_BETA, .tensor = beta.value().mesh_tensor()});
    }
    if (use_welford) {
        run_params.tensor_args.push_back(
            {.tensor_parameter_name = TP_RECIP, .tensor = tensor_args.recip_tensor->mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

CoreRangeSet LayerNormMultiCoreProgramFactory::default_core_range(IDevice* device) {
    auto grid_size = device->compute_with_storage_grid_size();
    return CoreRangeSet({CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1})});
}

}  // namespace ttnn::prim
