// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core.hpp"
#include <tt-metalium/circular_buffer_config.hpp>
#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <optional>
#include <bit>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

uint16_t bfloat16(float float_num) {
    uint32_t uint32_data;
    TT_FATAL(sizeof float_num == sizeof uint32_data, "sizeof data types not equal");

    uint32_data = *reinterpret_cast<uint32_t*>(&float_num);
    // just move upper 16 to lower 16 (truncate)
    uint32_data = (uint32_data >> 16);

    // store lower 16 as 16-bit uint
    return (uint16_t)uint32_data;
}

uint32_t pack_two_bfloat16_into_uint32(std::pair<uint16_t, uint16_t> two_bfloats) {
    // first -> lower 16
    // second -> upper 16
    return (uint32_t)two_bfloats.first | ((uint32_t)two_bfloats.second << 16);
}

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
    return sum < l1_size * 0.95;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

LayerNormMultiCoreProgramFactory::cached_program_t LayerNormMultiCoreProgramFactory::create(
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
    uint32_t Wp = padded_shape[-1], Hp = padded_shape[-2];
    uint32_t HWp = Hp * Wp;
    uint32_t NC = a.physical_volume() / HWp;

    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute

    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    //////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    IDevice* device = a.device();
    auto dst_addr = output.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                Circular Buffer Data Format Setup
    //////////////////////////////////////////////////////////////////////////
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    // Data span in tiles, rounded up to tile boundaries
    uint32_t Wt = Wp / TILE_WIDTH;
    uint32_t Ht = Hp / TILE_HEIGHT;

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

    auto a_addr = a.buffer()->address();
    auto b_dram_addr = b ? b.value().buffer()->address() : 0;
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;

    uint32_t num_tile_rows = NC * Ht;
    auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    // Create the sharded reciprocal LUT tensor if using Welford
    auto [recip_tensor, reciprocal_CB_size_bytes] =
        create_reciprocal_tensor_if_needed(device, W, all_cores, use_welford);

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
    uint32_t out0_t = block_size * 2;
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
    constexpr uint32_t with_weights_max_size = 56;
    constexpr uint32_t without_weights_max_size = 112;
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
        in2_t * bfloat16_tile_size,
        in3_t * bfloat16_tile_size,
        im2_t * single_tile_size,
        reciprocal_CB_size_bytes,
        a.device()->l1_size_per_core());
    if (!rms_norm and !use_row_major_kernel) {
        if ((gamma.has_value() or beta.has_value() or in_data_format == tt::DataFormat::Float32) and !cb_fits_in_L1) {
            // In the case that the required space is larger than what can be handeled by the single pass
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
    }

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
    Program program = CreateProgram();

    const auto use_welford_and_not_rms_norm = use_welford && !rms_norm;

    const auto fuse_pre_add = b.has_value();

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)block_size};
    if (!large_tensor_needed) {
        reader_compile_time_args.push_back((std::uint32_t)use_welford);
    }
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(b ? b->buffer() : nullptr).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(gamma ? gamma->buffer() : nullptr).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta ? beta->buffer() : nullptr).append_to(reader_compile_time_args);

    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().padded_shape()[-1] * gamma.value().element_size();
        reader_compile_time_args.push_back(gamma_stick_size);
    } else if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().padded_shape()[-1] * beta.value().element_size();
        reader_compile_time_args.push_back(beta_stick_size);
    } else {
        reader_compile_time_args.push_back(tile_size(datatype_to_dataformat_converter(a.dtype())));
    }

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)block_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> compute_defines;

    if (fuse_pre_add) {
        reader_defines["FUSE_PRE_ADD"] = "1";
        if (!use_welford) {
            compute_defines["FUSE_PRE_ADD"] = "1";
        }
    }

    if (gamma.has_value()) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_defines["FUSE_BETA"] = "1";
    }

    if (rms_norm) {
        compute_defines["RMSNORM"] = "1";
    }

    const auto* reader_kernel_path = use_row_major_kernel
                                         ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
                                           "reader_unary_interleaved_ln_rm_gb.cpp"
                                         : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
                                           "reader_unary_interleaved_ln.cpp";
    reader_kernel_path = large_tensor_needed
                             ? (use_welford_and_not_rms_norm
                                    ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
                                      "reader_unary_interleaved_ln_large_tensor_welford.cpp"
                                    : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
                                      "reader_unary_interleaved_ln_large_tensor.cpp")
                             : reader_kernel_path;

    auto reader_kernels_id = CreateKernel(
        program,
        reader_kernel_path,
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    bool float32_reduction = fp32_dest_acc_en && !legacy_reduction;
    std::vector<uint32_t> compute_args = {Wt, block_size, gamma.has_value(), beta.has_value(), fp32_dest_acc_en};
    if (use_welford_and_not_rms_norm) {
        compute_args.push_back(W);
        compute_args.push_back(ttnn::types::TILE_SIZE);
        compute_args.push_back(static_cast<uint32_t>(rms_norm));
        compute_args.push_back(static_cast<uint32_t>(fuse_pre_add));
    } else {
        compute_args.push_back(float32_reduction);
        compute_args.push_back(legacy_rsqrt);
        compute_args.push_back(W);
    }

    // The large-tensor non-Welford reduce kernel needs
    // an intermediate Float32 CB that can be unpacked
    // directly to dest (if doing a Float32 reduction)
    constexpr auto large_tensor_acc_cb = tt::CBIndex::c_26;
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (float32_reduction) {
        unpack_to_dest_mode[large_tensor_acc_cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    auto compute_kernels_id = CreateKernel(
        program,
        large_tensor_needed and !use_row_major_kernel
            ? (use_welford_and_not_rms_norm ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/"
                                              "layernorm_large_tensor_welford.cpp"
                                            : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/"
                                              "layernorm_large_tensor.cpp")
            : (use_welford_and_not_rms_norm
                   ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_welford.cpp"
                   : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp"),
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_args,
            .defines = compute_defines});

    // Create circular buffers
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(in0_t * in_single_tile_size, {{tt::CBIndex::c_0, in_data_format}})
            .set_page_size(tt::CBIndex::c_0, in_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_src0_config);
    CircularBufferConfig cb_out0_config =
        CircularBufferConfig(out0_t * out_single_tile_size, {{tt::CBIndex::c_16, out_data_format}})
            .set_page_size(tt::CBIndex::c_16, out_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out0_config);
    if (!rms_norm) {
        CircularBufferConfig cb_intermed1_config =
            CircularBufferConfig(im1_t * single_tile_size, {{tt::CBIndex::c_18, cb_data_format}})
                .set_page_size(tt::CBIndex::c_18, single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_intermed1_config);
    }
    if (!use_welford) {
        // Scaler for reduce
        CircularBufferConfig cb_in2_config =
            CircularBufferConfig(in2_t * bfloat16_tile_size, {{tt::CBIndex::c_2, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_2, bfloat16_tile_size);
        CreateCircularBuffer(program, all_cores, cb_in2_config);
    }
    CircularBufferConfig cb_in3_config =
        CircularBufferConfig(in3_t * bfloat16_tile_size, {{tt::CBIndex::c_3, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_3, bfloat16_tile_size);
    CreateCircularBuffer(program, all_cores, cb_in3_config);
    CircularBufferConfig cb_intermed2_config =
        CircularBufferConfig(im2_t * single_tile_size, {{tt::CBIndex::c_19, cb_data_format}})
            .set_page_size(tt::CBIndex::c_19, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed2_config);
    if (!(rms_norm && !b.has_value())) {
        CircularBufferConfig cb_intermed0_config =
            CircularBufferConfig(im0_t * single_tile_size, {{tt::CBIndex::c_24, cb_data_format}})
                .set_page_size(tt::CBIndex::c_24, single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_intermed0_config);
    }
    if (!use_welford) {
        CircularBufferConfig c_intermed3_config =
            CircularBufferConfig(im3_t * single_tile_size, {{tt::CBIndex::c_20, cb_data_format}})
                .set_page_size(tt::CBIndex::c_20, single_tile_size);
        CreateCircularBuffer(program, all_cores, c_intermed3_config);
    }
    CircularBufferConfig c_intermed4_config =
        CircularBufferConfig(im4_t * single_tile_size, {{tt::CBIndex::c_21, cb_data_format}})
            .set_page_size(tt::CBIndex::c_21, single_tile_size);
    CreateCircularBuffer(program, all_cores, c_intermed4_config);
    if (large_tensor_needed && !use_welford) {
        const auto large_tensor_acc_data_format = float32_reduction ? tt::DataFormat::Float32 : cb_data_format;
        const auto large_tensor_acc_tile_size = tt::tile_size(large_tensor_acc_data_format);
        CircularBufferConfig cb_large_tensor_acc_config =
            CircularBufferConfig(large_tensor_acc_tile_size, {{large_tensor_acc_cb, large_tensor_acc_data_format}})
                .set_page_size(large_tensor_acc_cb, large_tensor_acc_tile_size);
        CreateCircularBuffer(program, all_cores, cb_large_tensor_acc_config);
    }
    if (gamma.has_value() || beta.has_value()) {
        CircularBufferConfig c_intermed5_config =
            CircularBufferConfig(im5_t * single_tile_size, {{tt::CBIndex::c_22, cb_data_format}})
                .set_page_size(tt::CBIndex::c_22, single_tile_size);
        CreateCircularBuffer(program, all_cores, c_intermed5_config);
    }
    if (gamma.has_value()) {
        CircularBufferConfig c_in5_config =
            CircularBufferConfig(in5_t * gamma_single_tile_size, {{tt::CBIndex::c_5, gamma_cb_data_format}})
                .set_page_size(tt::CBIndex::c_5, gamma_single_tile_size);
        CreateCircularBuffer(program, all_cores, c_in5_config);
    }
    if (beta.has_value()) {
        CircularBufferConfig c_in6_config =
            CircularBufferConfig(in6_t * beta_single_tile_size, {{tt::CBIndex::c_6, beta_cb_data_format}})
                .set_page_size(tt::CBIndex::c_6, beta_single_tile_size);
        CreateCircularBuffer(program, all_cores, c_in6_config);
    }
    if (b) {
        // x = a+b in this notation
        // result = ln(x)*gamma + beta
        // if there's no pre-add we use cb_in0 for x, otherwise a is pre-buffered into in0, added into im6, then im6 is
        // used as x b is buffered into c_in1
        if (!rms_norm) {
            CircularBufferConfig c_intermed6_config =
                CircularBufferConfig(im6_t * single_tile_size, {{tt::CBIndex::c_23, cb_data_format}})
                    .set_page_size(tt::CBIndex::c_23, single_tile_size);
            CreateCircularBuffer(program, all_cores, c_intermed6_config);
        }
        // c_in1 is input buffer for b
        CircularBufferConfig c_in1_config =
            CircularBufferConfig(in1_t * inb_single_tile_size, {{tt::CBIndex::c_1, inb_data_format}})
                .set_page_size(tt::CBIndex::c_1, inb_single_tile_size);
        CreateCircularBuffer(program, all_cores, c_in1_config);
    }
    if (use_welford) {
        // Reciprocal LUT
        CircularBufferConfig c_recip_config =
            CircularBufferConfig(reciprocal_CB_size_bytes, {{tt::CBIndex::c_25, reciprocal_cb_data_format}})
                .set_page_size(tt::CBIndex::c_25, reciprocal_CB_size_bytes)
                .set_globally_allocated_address(*recip_tensor.value().buffer());
        CreateCircularBuffer(program, all_cores, c_recip_config);
    }

    uint32_t curr_row = 0;
    auto bfloat_one_value = bfloat16(1);
    uint32_t packed_one_value = pack_two_bfloat16_into_uint32({bfloat_one_value, bfloat_one_value});

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t tile_offset = curr_row * Wt;

        std::vector<uint32_t> reader_runtime_args = {
            a_addr,
            num_tile_rows_per_core,
            Wt,
            tile_offset,
            packed_one_value,
            std::bit_cast<uint32_t>(eps),
            gamma_dram_addr,
            beta_dram_addr,
            b_dram_addr};
        if (!(use_welford && large_tensor_needed)) {
            reader_runtime_args.push_back(W);
        }
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);
        SetRuntimeArgs(program, compute_kernels_id, core, {num_tile_rows_per_core});
        SetRuntimeArgs(program, writer_kernels_id, core, {dst_addr, Wt, num_tile_rows_per_core, tile_offset});
        curr_row += num_tile_rows_per_core;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = reader_kernels_id,
            .writer_kernel_id = writer_kernels_id,
            .num_cores = num_cores,
            .grid_size = grid_size}};
}

void LayerNormMultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const LayerNormParams& /*operation_attributes*/,
    const LayerNormInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* const src_a_dram_buffer = tensor_args.input.buffer();
    const auto& src_b_tensor = tensor_args.residual_input_tensor;
    const auto& gamma_tensor = tensor_args.weight;
    const auto& beta_tensor = tensor_args.bias;
    auto* const dst_dram_buffer = tensor_return_value.buffer();

    auto* src_b_dram_buffer = src_b_tensor.has_value() ? src_b_tensor.value().buffer() : nullptr;
    auto* gamma_dram_buffer = gamma_tensor.has_value() ? gamma_tensor.value().buffer() : nullptr;
    auto* beta_dram_buffer = beta_tensor.has_value() ? beta_tensor.value().buffer() : nullptr;

    const auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    for (uint32_t i = 0; i < shared_vars.num_cores; ++i) {
        CoreCoord core = {i % shared_vars.grid_size.x, i / shared_vars.grid_size.x};

        {
            auto& runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_id, core);
            runtime_args[0] = src_a_dram_buffer->address();
            if (src_b_dram_buffer != nullptr) {
                runtime_args[8] = src_b_dram_buffer->address();
            }
            if (gamma_dram_buffer != nullptr) {
                runtime_args[6] = gamma_dram_buffer->address();
            }
            if (beta_dram_buffer != nullptr) {
                runtime_args[7] = beta_dram_buffer->address();
            }
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, shared_vars.writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
