// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <optional>
#include <variant>

#include "cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_post_all_gather_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "tt-metalium/circular_buffer_config.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/circular_buffer.hpp>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::normalization {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
inline bool is_dram(const Tensor& input_tensor) {
    return input_tensor.memory_config().buffer_type() == BufferType::DRAM;
}
inline bool is_dram(const std::optional<const Tensor>& input_tensor) {
    return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

inline uint16_t bfloat16(float float_num) {
    uint32_t uint32_data;
    TT_FATAL(
        sizeof float_num == sizeof uint32_data,
        "Float size ({}) must equal uint32 size ({})",
        sizeof float_num,
        sizeof uint32_data);

    uint32_data = *reinterpret_cast<uint32_t*>(&float_num);
    // just move upper 16 to lower 16 (truncate)
    uint32_data = (uint32_data >> 16);

    // store lower 16 as 16-bit uint
    return (uint16_t)uint32_data;
}
inline uint32_t pack_two_bfloat16_into_uint32(std::pair<uint16_t, uint16_t> two_bfloats) {
    // first -> lower 16
    // second -> upper 16
    return (uint32_t)two_bfloats.first | ((uint32_t)two_bfloats.second << 16);
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

// computes layernorm(a)*gamma + beta
tt::tt_metal::operation::ProgramWithCallbacks layernorm_post_allgather_multi_core(
    const Tensor& a,
    const Tensor& stats,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    LayerNormDistributedType norm_type,
    float eps,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using tt::tt_metal::CBHandle;
    using tt::tt_metal::CircularBuffer;
    using tt::tt_metal::CircularBufferConfig;

    const bool is_rmsnorm = norm_type == LayerNormDistributedType::RMSNORM;
    const auto shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute
    const auto& a_dtype = a.dtype();

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t stats_tiles_cols = stats.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t tile_cols_per_device = is_rmsnorm ? 1 : 2;
    const uint32_t num_devices = stats_tiles_cols / tile_cols_per_device;
    TT_FATAL(num_devices > 0, "Number of devices must be greater than 0");
    TT_FATAL(
        num_devices * tile_cols_per_device == stats_tiles_cols, "Number of devices must divide number of stats tiles");

    uint32_t num_tile_rows = NC * Ht;

    log_debug(tt::LogOp, "is_rmsnorm: {}", is_rmsnorm);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "H: {}", H);
    log_debug(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "Ht: {}", Ht);
    log_debug(tt::LogOp, "stats_tiles_cols: {}", stats_tiles_cols);
    log_debug(tt::LogOp, "num_devices: {}", num_devices);

    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    //////////////////////////////////////////////////////////////////////////
    IDevice* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                Circular Buffer Data Format Setup
    //////////////////////////////////////////////////////////////////////////
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t block_size =
        fp32_dest_acc_en ? tt::tt_metal::find_max_divisor(Wt, 4) : tt::tt_metal::find_max_divisor(Wt, 8);

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat stats_data_format = tt::tt_metal::datatype_to_dataformat_converter(stats.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format = gamma.has_value()
                                              ? tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype())
                                              : tt::DataFormat::Float16_b;
    tt::DataFormat beta_cb_data_format = beta.has_value()
                                             ? tt::tt_metal::datatype_to_dataformat_converter(beta.value().dtype())
                                             : tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tt_metal::detail::TileSize(in_data_format);
    uint32_t stats_single_tile_size = tt::tt_metal::detail::TileSize(stats_data_format);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t out_single_tile_size = tt::tt_metal::detail::TileSize(out_data_format);
    uint32_t bfloat16_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    uint32_t gamma_single_tile_size = tt::tt_metal::detail::TileSize(gamma_cb_data_format);
    uint32_t beta_single_tile_size = tt::tt_metal::detail::TileSize(beta_cb_data_format);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "gamma_cb_data_format: {}", gamma_cb_data_format);
    log_debug(tt::LogOp, "beta_cb_data_format: {}", beta_cb_data_format);
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);

    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;

    auto a_addr = a.buffer()->address();
    auto stats_addr = stats.buffer()->address();
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    auto dst_addr = output.buffer()->address();

    uint32_t num_tiles = a.physical_volume() / TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().physical_volume() / TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().physical_volume() / TILE_HW : 0;

    // For bert, tensor is packed as RM with width 32
    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        num_gamma_tiles = gamma.has_value() ? gamma.value().physical_volume() / TILE_WIDTH : 0;
    }
    if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        num_beta_tiles = beta.has_value() ? beta.value().physical_volume() / TILE_WIDTH : 0;
    }

    log_debug(tt::LogOp, "num_gamma_tiles: {}", num_gamma_tiles);
    log_debug(tt::LogOp, "num_beta_tiles: {}", num_beta_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    /*
    in0_cb: a
    in1_cb: stats
    in2_cb: gamma
    in3_cb: beta
    in4_cb: epsilon
    in5_cb: 1/row_size (reduction scalar)

    intermediate CBs are packed such that in layernorm, first tile is for x**2 stats, second tile is for x stats
    in RMSNorm, only first tile has valid data.

    intermed0_cb: [mean(x**2), mean(x)] # reduce with reduce_scalar
    intermed1_cb: mean(x)**2 # LN only
    intermed2_cb: var = mean(x**2) - mean(x)**2 # for RMSNorm, this is just mean(x**2)
    intermed3_cb: var + epsilon # RMSNorm takes mean(x**2) instead of var
    intermed4_cb: 1/sqrt(var + epsilon)
    intermed5_cb: x - mean(x) # LN only
    intermed6_cb: (x - mean(x)) * 1/sqrt(var + epsilon) # RMSNorm takes x instead of (x - mean(x))
    intermed7_cb: (x - mean(x)) * 1/sqrt(var + epsilon) * gamma
    out0_cb: (x - mean(x)) * 1/sqrt(var + epsilon) * gamma + beta # RMSNorm doesn't include beta

    */

    const uint32_t in0_tiles = Wt;
    const uint32_t in1_tiles = stats_tiles_cols;
    const uint32_t in2_tiles = Wt;
    const uint32_t in3_tiles = Wt;
    const uint32_t in4_tiles = 1;  // epsilon
    const uint32_t in5_tiles = 1;  // reduce scalar

    const uint32_t intermed0_tiles = tile_cols_per_device;
    const uint32_t intermed1_tiles = 1;
    const uint32_t intermed2_tiles = 1;
    const uint32_t intermed3_tiles = 1;
    const uint32_t intermed4_tiles = 1;
    const uint32_t intermed5_tiles = Wt;
    const uint32_t intermed6_tiles = Wt;
    const uint32_t intermed7_tiles = Wt;
    const uint32_t out0_tiles = Wt;

    TT_FATAL(
        W <= TILE_WIDTH * in0_tiles,
        "W ({}) exceeds the maximum supported size of tile buffer ({} * {}, kernel limitation right now)",
        W,
        TILE_WIDTH,
        in0_tiles);
    TT_FATAL(
        in0_tiles % block_size == 0,
        "Buffer size in0_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        in0_tiles,
        block_size);
    TT_FATAL(
        in2_tiles % block_size == 0,
        "Buffer size in2_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        in2_tiles,
        block_size);
    TT_FATAL(
        in3_tiles % block_size == 0,
        "Buffer size in3_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        in3_tiles,
        block_size);
    TT_FATAL(
        out0_tiles % block_size == 0,
        "Buffer size out0_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        out0_tiles,
        block_size);
    TT_FATAL(
        intermed5_tiles % block_size == 0,
        "Buffer size im0_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        intermed5_tiles,
        block_size);
    TT_FATAL(
        intermed6_tiles % block_size == 0,
        "Buffer size im6_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        intermed6_tiles,
        block_size);
    TT_FATAL(
        intermed7_tiles % block_size == 0,
        "Buffer size im7_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        intermed7_tiles,
        block_size);

    auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "grid_size: {}", grid_size);
    log_debug(tt::LogOp, "core_group_1: {}", core_group_1.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_1: {}", num_tile_rows_per_core_group_1);
    log_debug(tt::LogOp, "core_group_2: {}", core_group_2.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_2: {}", num_tile_rows_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = tt::tt_metal::CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)is_dram(a),
        (std::uint32_t)is_dram(stats),
        (std::uint32_t)is_dram(gamma),
        (std::uint32_t)is_dram(beta),
        (std::uint32_t)block_size,
        (std::uint32_t)stats_tiles_cols,
    };

    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().padded_shape()[-1] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(gamma_stick_size);
        TT_FATAL(gamma_stick_size_is_power_of_two, "Only power of 2 gammas are supported");
        reader_compile_time_args.push_back((std::uint32_t)gamma_stick_size_is_power_of_two);
        // if (gamma_stick_size_is_power_of_two) {
        uint32_t gamma_log2_stick_size =
            gamma_stick_size_is_power_of_two ? (std::uint32_t)std::log2(gamma_stick_size) : 0;
        reader_compile_time_args.push_back((std::uint32_t)gamma_log2_stick_size);
    }

    std::vector<uint32_t> writer_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)is_dram(output),
                                                      (std::uint32_t)block_size};

    bool tile_dtype_is_bfloat16 = a.dtype() == tt::tt_metal::DataType::BFLOAT16;
    std::map<string, string> reader_defines;
    std::map<string, string> compute_defines;
    if (gamma.has_value()) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_defines["FUSE_BETA"] = "1";
    }

    if (is_rmsnorm) {
        compute_defines["RMSNORM"] = "1";
    }

    auto use_row_major_kernel = (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) or
                                (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR);
    TT_FATAL(
        use_row_major_kernel || (!gamma.has_value() && !beta.has_value()),
        "Only row major gamma and beta are supported");
    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_unary_interleaved_ln_rm_gb_post_allgather.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args = {
        Wt, block_size, stats_tiles_cols, gamma.has_value(), beta.has_value(), fp32_dest_acc_en};

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
        "layernorm_post_allgather.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_args,
            .defines = compute_defines});

    // Create circular buffers
    // c_in0 -> a
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(in0_tiles * in_single_tile_size, {{tt::CBIndex::c_0, in_data_format}})
            .set_page_size(tt::CBIndex::c_0, in_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_src0_config);
    // c_in1 -> stats
    CircularBufferConfig cb_stats_config =
        CircularBufferConfig(in1_tiles * stats_single_tile_size, {{tt::CBIndex::c_1, stats_data_format}})
            .set_page_size(tt::CBIndex::c_1, stats_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_stats_config);
    // c_in2 -> gamma
    if (gamma.has_value()) {
        CircularBufferConfig cb_gamma_config =
            CircularBufferConfig(in2_tiles * gamma_single_tile_size, {{tt::CBIndex::c_2, gamma_cb_data_format}})
                .set_page_size(tt::CBIndex::c_2, gamma_single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_gamma_config);
    }
    // c_in3 -> beta
    if (beta.has_value()) {
        CircularBufferConfig cb_beta_config =
            CircularBufferConfig(in3_tiles * beta_single_tile_size, {{tt::CBIndex::c_3, beta_cb_data_format}})
                .set_page_size(tt::CBIndex::c_3, beta_single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_beta_config);
    }
    // c_in4 -> epsilon
    CircularBufferConfig cb_eps_config =
        CircularBufferConfig(in4_tiles * bfloat16_tile_size, {{tt::CBIndex::c_4, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_4, bfloat16_tile_size);
    CreateCircularBuffer(program, all_cores, cb_eps_config);
    // c_in5 -> reduce scalar
    CircularBufferConfig cb_reduce_config =
        CircularBufferConfig(in5_tiles * bfloat16_tile_size, {{tt::CBIndex::c_5, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_5, bfloat16_tile_size);
    CreateCircularBuffer(program, all_cores, cb_reduce_config);

    // LN and RMS shared intermediates //
    // c_intermed0 -> [mean(x**2), mean(x)]
    CircularBufferConfig cb_intermed0_config =
        CircularBufferConfig(intermed0_tiles * single_tile_size, {{tt::CBIndex::c_6, cb_data_format}})
            .set_page_size(tt::CBIndex::c_6, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed0_config);
    // c_intermed2 -> var = mean(x**2) - mean(x)**2
    CircularBufferConfig cb_intermed2_config =
        CircularBufferConfig(intermed2_tiles * single_tile_size, {{tt::CBIndex::c_8, cb_data_format}})
            .set_page_size(tt::CBIndex::c_8, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed2_config);
    // c_intermed3 -> var + epsilon
    CircularBufferConfig cb_intermed3_config =
        CircularBufferConfig(intermed3_tiles * single_tile_size, {{tt::CBIndex::c_9, cb_data_format}})
            .set_page_size(tt::CBIndex::c_9, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed3_config);
    // c_intermed4 -> 1/sqrt(var + epsilon)
    CircularBufferConfig cb_intermed4_config =
        CircularBufferConfig(intermed4_tiles * single_tile_size, {{tt::CBIndex::c_10, cb_data_format}})
            .set_page_size(tt::CBIndex::c_10, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed4_config);
    // c_intermed6 -> (x - mean(x)) * 1/sqrt(var + epsilon)
    CircularBufferConfig cb_intermed6_config =
        CircularBufferConfig(intermed6_tiles * single_tile_size, {{tt::CBIndex::c_12, cb_data_format}})
            .set_page_size(tt::CBIndex::c_12, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed6_config);

    // LN-specific intermediates
    if (!is_rmsnorm) {
        // c_intermed1 -> mean(x)**2
        CircularBufferConfig cb_intermed1_config =
            CircularBufferConfig(intermed1_tiles * single_tile_size, {{tt::CBIndex::c_7, cb_data_format}})
                .set_page_size(tt::CBIndex::c_7, single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_intermed1_config);
        // c_intermed5 -> x - mean(x)
        CircularBufferConfig cb_intermed5_config =
            CircularBufferConfig(intermed5_tiles * single_tile_size, {{tt::CBIndex::c_11, cb_data_format}})
                .set_page_size(tt::CBIndex::c_11, single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_intermed5_config);
        if (beta.has_value()) {
            // Layernorm has gamma and beta so we need an extra intermediate buffer
            // c_intermed7 -> (x - mean(x)) * 1/sqrt(var + epsilon) * gamma
            CircularBufferConfig cb_intermed7_config =
                CircularBufferConfig(intermed7_tiles * single_tile_size, {{tt::CBIndex::c_13, cb_data_format}})
                    .set_page_size(tt::CBIndex::c_13, single_tile_size);
            CreateCircularBuffer(program, all_cores, cb_intermed7_config);
        }
    }

    CircularBufferConfig cb_out0_config =
        CircularBufferConfig(out0_tiles * out_single_tile_size, {{tt::CBIndex::c_14, out_data_format}})
            .set_page_size(tt::CBIndex::c_14, out_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out0_config);

    // Log all circular buffers with program.circular_buffers(), which returns
    // std::vector<std::shared_ptr<CircularBuffer>>

    for (const auto& cb : program.circular_buffers()) {
        for (const auto index : cb->buffer_indices()) {
            log_debug(tt::LogOp, "cb_id {}", index);
            log_debug(tt::LogOp, "page_size: {}", cb->page_size(index));
            log_debug(tt::LogOp, "num_pages: {}", cb->num_pages(index));
            log_debug(tt::LogOp, "data_format: {}", cb->data_format(index));
        }
    }

    uint32_t curr_row = 0;
    float winv = 1.0f / (W * num_devices);  // bcast-w scaler
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union {
        float f;
        uint32_t u;
    } e;
    e.f = eps;  // epsilon
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
        uint32_t stats_offset = curr_row * stats_tiles_cols;

        SetRuntimeArgs(
            program,
            reader_kernels_id,
            core,
            {a_addr,
             num_tile_rows_per_core,
             Wt,
             tile_offset,
             stats_offset,
             packed_winv_value,
             e.u,  // 0-5
             gamma_dram_addr,
             beta_dram_addr,
             stats_addr}  // 6-8
        );
        SetRuntimeArgs(program, compute_kernels_id, core, {num_tile_rows_per_core});
        SetRuntimeArgs(program, writer_kernels_id, core, {dst_addr, num_tile_rows_per_core * Wt, tile_offset});
        curr_row += num_tile_rows_per_core;
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_id = reader_kernels_id, writer_kernel_id = writer_kernels_id, num_cores, grid_size](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input_tensor = input_tensors.at(0);
            const auto& stats_tensor = input_tensors.at(1);
            const auto& gamma_tensor = optional_input_tensors.at(0);
            const auto& beta_tensor = optional_input_tensors.at(1);

            const auto input_addr = input_tensor.buffer()->address();
            const auto stats_addr = stats_tensor.buffer()->address();
            const bool has_gamma = gamma_tensor.has_value();
            const bool has_beta = beta_tensor.has_value();
            const auto gamma_addr = has_gamma ? gamma_tensor.value().buffer()->address() : 0;
            const auto beta_addr = has_beta ? beta_tensor.value().buffer()->address() : 0;

            const auto& output_tensor = output_tensors.at(0);
            const auto output_addr = output_tensor.buffer()->address();

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);

            for (uint32_t i = 0; i < num_cores; ++i) {
                const CoreCoord core = {i % grid_size.x, i / grid_size.x};

                {
                    auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);

                    reader_args[0] = input_addr;
                    reader_args[9] = stats_addr;
                    if (has_gamma) {
                        reader_args[7] = gamma_addr;
                    }
                    if (has_beta) {
                        reader_args[8] = beta_addr;
                    }
                }

                {
                    auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
                    writer_args[0] = output_addr;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::normalization
