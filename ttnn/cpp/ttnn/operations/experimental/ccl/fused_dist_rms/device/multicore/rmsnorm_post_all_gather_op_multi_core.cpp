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
    const Tensor& input_tensor,
    const Tensor& stats_tensor,
    Tensor& output_tensor,
    const std::optional<const Tensor>& weight_tensor,
    float eps,
    uint32_t num_heads,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    Program program = tt::tt_metal::CreateProgram();

    const bool has_weight = weight_tensor.has_value();

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t W = input_shape[-1];
    const uint32_t H = input_shape[-2];

    const uint32_t num_tile_cols = W / TILE_WIDTH;
    const uint32_t num_tile_rows = H / TILE_HEIGHT;
    const uint32_t head_dim_tiles = num_tile_cols / num_heads;

    const uint32_t stats_tiles_cols = stats_tensor.padded_shape()[-1] / TILE_WIDTH;
    // AllGather results in a tensor with num_devices columns of tiles
    const uint32_t num_devices = stats_tiles_cols;
    TT_FATAL(num_devices > 0, "Number of devices must be greater than 0");

    log_info(tt::LogOp, "W: {}", W);
    log_info(tt::LogOp, "H: {}", H);
    log_info(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_info(tt::LogOp, "num_tile_cols: {}", num_tile_cols);
    log_info(tt::LogOp, "stats_tiles_cols: {}", stats_tiles_cols);
    log_info(tt::LogOp, "num_devices: {}", num_devices);
    log_info(tt::LogOp, "has_weight: {}", has_weight);

    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    //////////////////////////////////////////////////////////////////////////
    IDevice* device = input_tensor.device();
    const auto grid_size = device->compute_with_storage_grid_size();
    const auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    const uint32_t num_cores = core_grid.size();

    ////////////////////////////////////////////////////////////////////////////
    //                Circular Buffer Data Format Setup
    //////////////////////////////////////////////////////////////////////////
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const uint32_t dst_reg_count = get_dest_reg_count(compute_kernel_config);

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat stats_data_format = tt::tt_metal::datatype_to_dataformat_converter(stats_tensor.dtype());
    tt::DataFormat intermediate_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat reduce_scalar_data_format = tt::DataFormat::Float16_b;

    tt::DataFormat weight_data_format =
        has_weight ? tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.value().dtype())
                   : tt::DataFormat::Float16_b;

    uint32_t input_tile_size = tt::tile_size(input_data_format);
    uint32_t output_tile_size = tt::tile_size(output_data_format);
    uint32_t intermediate_tile_size = tt::tile_size(intermediate_data_format);
    uint32_t reduce_scalar_tile_size = tt::tile_size(reduce_scalar_data_format);
    uint32_t stats_tile_size = tt::tile_size(stats_data_format);
    uint32_t weight_tile_size = tt::tile_size(weight_data_format);

    log_info(tt::LogOp, "input_data_format: {}", input_data_format);
    log_info(tt::LogOp, "output_data_format: {}", output_data_format);
    log_info(tt::LogOp, "stats_data_format: {}", stats_data_format);
    log_info(tt::LogOp, "intermediate_data_format: {}", intermediate_data_format);
    log_info(tt::LogOp, "reduce_scalar_data_format: {}", reduce_scalar_data_format);
    log_info(tt::LogOp, "weight_data_format: {}", weight_data_format);

    auto input_addr = input_tensor.buffer()->address();
    auto output_addr = output_tensor.buffer()->address();
    auto stats_addr = stats_tensor.buffer()->address();
    auto weight_addr = has_weight ? weight_tensor.value().buffer()->address() : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    //////////////////////////////////////////////////////////////////////////
    /*
    CB 0: input
    CB 1: stats
    CB 2: reduce scalar
    CB 3: epsilon
    CB 4: reduce result
    CB 5: output
    CB 6: weight
    CB 7: intermediate hold x * RMS before applying weight
    */

    const uint32_t input_cb_id = tt::CBIndex::c_0;
    const uint32_t stats_cb_id = tt::CBIndex::c_1;
    const uint32_t reduce_scalar_cb_id = tt::CBIndex::c_2;
    const uint32_t epsilon_cb_id = tt::CBIndex::c_3;
    const uint32_t reduce_result_cb_id = tt::CBIndex::c_4;
    const uint32_t output_cb_id = tt::CBIndex::c_5;
    const uint32_t weight_cb_id = tt::CBIndex::c_6;
    const uint32_t intermediate_cb_id = tt::CBIndex::c_7;

    constexpr uint32_t double_buffer_constant = 2;
    const uint32_t input_cb_num_tiles = dst_reg_count * double_buffer_constant;
    const uint32_t stats_cb_num_tiles = stats_tiles_cols * double_buffer_constant;
    const uint32_t reduce_scalar_cb_num_tiles = 1;
    const uint32_t epsilon_cb_num_tiles = 1;
    const uint32_t reduce_result_cb_num_tiles = 1;
    const uint32_t output_cb_num_tiles = dst_reg_count * double_buffer_constant;
    const uint32_t weight_cb_num_tiles = num_tile_cols;
    const uint32_t intermediate_cb_num_tiles = dst_reg_count;

    create_cb(input_cb_id, program, core_grid, input_tile_size, input_cb_num_tiles, input_data_format);

    create_cb(stats_cb_id, program, core_grid, stats_tile_size, stats_cb_num_tiles, stats_data_format);

    create_cb(
        reduce_scalar_cb_id,
        program,
        core_grid,
        reduce_scalar_tile_size,
        reduce_scalar_cb_num_tiles,
        reduce_scalar_data_format);

    create_cb(
        epsilon_cb_id, program, core_grid, reduce_scalar_tile_size, epsilon_cb_num_tiles, reduce_scalar_data_format);

    create_cb(
        reduce_result_cb_id,
        program,
        core_grid,
        intermediate_tile_size,
        reduce_result_cb_num_tiles,
        intermediate_data_format);

    create_cb(output_cb_id, program, core_grid, output_tile_size, output_cb_num_tiles, output_data_format);

    if (has_weight) {
        create_cb(weight_cb_id, program, core_grid, weight_tile_size, weight_cb_num_tiles, weight_data_format);
        create_cb(
            intermediate_cb_id,
            program,
            core_grid,
            intermediate_tile_size,
            intermediate_cb_num_tiles,
            intermediate_data_format);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    float winv = 1.0f / (W * num_devices);  // bcast-w scaler
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union {
        float f;
        uint32_t u;
    } e{};
    e.f = eps;  // epsilon

    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_id,
        stats_cb_id,
        weight_cb_id,
        reduce_scalar_cb_id,
        epsilon_cb_id,
        num_tile_cols,
        dst_reg_count,
        stats_tiles_cols,
        packed_winv_value,
        e.u,
        has_weight,
    };

    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(stats_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(has_weight ? weight_tensor.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_id,
        num_tile_cols,
        dst_reg_count,
        head_dim_tiles,
        num_heads,
        num_tile_rows,
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/fused_dist_rms/device/kernels/dataflow/"
        "rms_post_allgather_reader.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/fused_dist_rms/device/kernels/dataflow/"
        "rms_post_allgather_writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    bool use_float32_reduction = fp32_dest_acc_en;  // legacy_reduction=false
    bool use_legacy_rsqrt = false;
    std::vector<uint32_t> compute_args = {
        input_cb_id,
        stats_cb_id,
        weight_cb_id,
        reduce_scalar_cb_id,
        epsilon_cb_id,
        reduce_result_cb_id,
        intermediate_cb_id,
        output_cb_id,
        num_tile_cols,
        dst_reg_count,
        stats_tiles_cols,
        use_float32_reduction,
        use_legacy_rsqrt,
        has_weight,
    };

    auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/ccl/fused_dist_rms/device/kernels/compute/rmsnorm_post_allgather.cpp";
    auto compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_args};
    auto compute_kernels_id = CreateKernel(program, compute_kernel_file, core_grid, compute_config);

    const uint32_t num_tile_rows_per_core = tt::div_up(num_tile_rows, num_cores);

    const auto cores = corerange_to_cores(core_grid, num_cores, true);
    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);

        const uint32_t tile_row_start = std::min(core_id * num_tile_rows_per_core, num_tile_rows);
        const uint32_t tile_row_end = std::min(tile_row_start + num_tile_rows_per_core, num_tile_rows);
        const uint32_t num_tile_rows_to_process = tile_row_end - tile_row_start;

        std::vector<uint32_t> reader_runtime_args = {
            input_addr,
            stats_addr,
            weight_addr,
            tile_row_start,
            tile_row_end,
        };
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        std::vector<uint32_t> compute_runtime_args = {num_tile_rows_to_process};
        SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);

        std::vector<uint32_t> writer_runtime_args = {
            output_addr,
            tile_row_start,
            tile_row_end,
        };
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);
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
            const auto& weight_tensor = optional_input_tensors.at(0);

            const auto input_addr = input_tensor.buffer()->address();
            const auto stats_addr = stats_tensor.buffer()->address();
            const auto weight_addr = weight_tensor.has_value() ? weight_tensor.value().buffer()->address() : 0;
            const auto& output_tensor = output_tensors.at(0);
            const auto output_addr = output_tensor.buffer()->address();

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);

            for (const auto& core : cores) {
                {
                    auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);
                    reader_args[0] = input_addr;
                    reader_args[1] = stats_addr;
                    reader_args[2] = weight_addr;
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
