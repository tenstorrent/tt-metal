// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <string>
#include <variant>

#include "ttnn/operations/experimental/ccl/fused_dist_rms/device/rmsnorm_post_all_gather_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "tt-metalium/circular_buffer_config.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::experimental::ccl {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

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
tt::tt_metal::operation::ProgramWithCallbacks fused_rmsnorm_post_allgather_multi_core(
    const Tensor& a,
    const Tensor& stats,
    Tensor& output,
    float eps,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using tt::tt_metal::CBHandle;
    using tt::tt_metal::CircularBuffer;
    using tt::tt_metal::CircularBufferConfig;

    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t stats_tiles_cols = stats.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t tile_cols_per_device = 1;
    const uint32_t num_devices = stats_tiles_cols / tile_cols_per_device;
    TT_FATAL(num_devices > 0, "Number of devices must be greater than 0");
    TT_FATAL(
        num_devices * tile_cols_per_device == stats_tiles_cols, "Number of devices must divide number of stats tiles");

    uint32_t num_tile_rows = NC * Ht;

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

    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t stats_single_tile_size = tt::tile_size(stats_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);

    auto a_addr = a.buffer()->address();
    auto stats_addr = stats.buffer()->address();
    auto dst_addr = output.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    //////////////////////////////////////////////////////////////////////////
    /*
    in0_cb: a
    in1_cb: stats
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
    const uint32_t in4_tiles = 1;  // epsilon
    const uint32_t in5_tiles = 1;  // reduce scalar

    const uint32_t intermed0_tiles = tile_cols_per_device;
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
    // no beta buffer
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
    auto cores = corerange_to_cores(all_cores, std::nullopt);
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = tt::tt_metal::CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)block_size,
        (std::uint32_t)stats_tiles_cols,
    };

    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(stats.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)block_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> compute_defines;

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/fused_dist_rms/device/kernels/dataflow/"
        "rme_post_allgather_reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/fused_dist_rms/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    bool float32_reduction = fp32_dest_acc_en;  // legacy_reduction=false
    uint32_t tiles_per_core_y = Wt;
    std::vector<uint32_t> compute_args = {
        tiles_per_core_y,
        block_size,
        stats_tiles_cols,
        false,  // has gamma
        false,  // has beta
        fp32_dest_acc_en,
        float32_reduction ? 1 : 0,
        0};

    auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/fused_dist_rms/device/kernels/compute/rmsnorm_post_allgather.cpp";
    auto compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_args,
        .defines = compute_defines};
    auto compute_kernels_id = CreateKernel(program, compute_kernel_file, all_cores, compute_config);

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

    // c_in3 -> beta
    // No beta buffer in RMSNorm
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

    CircularBufferConfig cb_out0_config =
        CircularBufferConfig(out0_tiles * out_single_tile_size, {{tt::CBIndex::c_14, out_data_format}})
            .set_page_size(tt::CBIndex::c_14, out_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out0_config);

    // Log all circular buffers with program.circular_buffers(), which returns
    // std::vector<std::shared_ptr<CircularBuffer>>

    for (const auto& cb : program.circular_buffers()) {
        for ([[maybe_unused]] const auto index : cb->buffer_indices()) {
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
    } e{};
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
        uint32_t y_offset = 0;

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
             0,    // gamma
             0,
             stats_addr,
             y_offset}  // 6-8
        );
        SetRuntimeArgs(program, compute_kernels_id, core, {num_tile_rows_per_core});
        SetRuntimeArgs(program, writer_kernels_id, core, {dst_addr, num_tile_rows_per_core * Wt, tile_offset});
        curr_row += num_tile_rows_per_core;
    }
    auto override_runtime_arguments_callback =
        [reader_kernel_id = reader_kernels_id, writer_kernel_id = writer_kernels_id, cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input_tensor = input_tensors.at(0);
            const auto& stats_tensor = input_tensors.at(1);

            const auto input_addr = input_tensor.buffer()->address();
            const auto stats_addr = stats_tensor.buffer()->address();

            const auto& output_tensor = output_tensors.at(0);
            const auto output_addr = output_tensor.buffer()->address();

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);

            for (const auto& core : cores) {
                {
                    auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);

                    reader_args[0] = input_addr;
                    reader_args[9] = stats_addr;
                }

                {
                    auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
                    writer_args[0] = output_addr;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::ccl
