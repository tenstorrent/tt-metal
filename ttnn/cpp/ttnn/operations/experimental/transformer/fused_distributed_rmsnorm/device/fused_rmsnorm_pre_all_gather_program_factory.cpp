// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_rmsnorm_pre_all_gather_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include <optional>
#include <string>
#include <variant>

namespace ttnn::experimental::prim {

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

FusedRMSNormPreAllGatherProgramFactory::cached_program_t FusedRMSNormPreAllGatherProgramFactory::create(
    const FusedRmsnormPreAllGatherParams& operation_attributes,
    const FusedRmsnormPreAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    Program program = tt::tt_metal::CreateProgram();

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t W = input_shape[-1];
    const uint32_t folded_H = input_tensor.physical_volume() / W;

    const uint32_t num_tile_cols = W / TILE_WIDTH;
    const uint32_t num_tile_rows = folded_H / TILE_HEIGHT;
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "folded_H: {}", folded_H);
    log_debug(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_debug(tt::LogOp, "num_tile_cols: {}", num_tile_cols);

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
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const uint32_t dst_reg_count = get_dest_reg_count(compute_kernel_config);

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat reduce_scalar_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat intermediate_data_format = tt::DataFormat::Float32;
    uint32_t input_tile_size = tt::tile_size(input_data_format);
    uint32_t output_tile_size = tt::tile_size(output_data_format);
    uint32_t intermediate_tile_size = tt::tile_size(intermediate_data_format);
    uint32_t reduce_scalar_tile_size = tt::tile_size(reduce_scalar_data_format);

    log_debug(tt::LogOp, "input_data_format: {}", input_data_format);
    log_debug(tt::LogOp, "output_data_format: {}", output_data_format);
    log_debug(tt::LogOp, "intermediate_data_format: {}", intermediate_data_format);
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);
    log_debug(tt::LogOp, "dst_reg_count: {}", dst_reg_count);

    auto input_addr = input_tensor.buffer()->address();
    auto output_addr = output_tensor.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    /*
    CB 0: input
    CB 1: reduce scalar
    CB 2: partial sum
    CB 3: output
    */
    const uint32_t output_tiles_per_row = 1;

    const uint32_t double_buffer_constant = 2;
    const uint32_t input_cb_num_tiles = dst_reg_count * double_buffer_constant;
    const uint32_t reduce_scalar_cb_num_tiles = 1;
    const uint32_t intermediate_cb_num_tiles = 1;
    const uint32_t output_cb_num_tiles = output_tiles_per_row * double_buffer_constant;

    const uint32_t num_tile_rows_per_core = tt::div_up(num_tile_rows, num_cores);

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "grid_size: {}", grid_size);
    log_debug(tt::LogOp, "core_grid: {}", core_grid);
    log_debug(tt::LogOp, "num_tile_rows_per_core: {}", num_tile_rows_per_core);

    const uint32_t input_cb_id = tt::CBIndex::c_0;
    const uint32_t reduce_scalar_cb_id = tt::CBIndex::c_1;
    const uint32_t intermediate_cb_id = tt::CBIndex::c_2;
    const uint32_t output_cb_id = tt::CBIndex::c_3;

    tt::tt_metal::create_cb(input_cb_id, program, core_grid, input_tile_size, input_cb_num_tiles, input_data_format);

    tt::tt_metal::create_cb(
        reduce_scalar_cb_id,
        program,
        core_grid,
        reduce_scalar_tile_size,
        reduce_scalar_cb_num_tiles,
        reduce_scalar_data_format);

    tt::tt_metal::create_cb(
        intermediate_cb_id,
        program,
        core_grid,
        intermediate_tile_size,
        intermediate_cb_num_tiles,
        intermediate_data_format);

    tt::tt_metal::create_cb(
        output_cb_id, program, core_grid, output_tile_size, output_cb_num_tiles, output_data_format);

    float winv = 1.0f;
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_id,
        reduce_scalar_cb_id,
        num_tile_cols,
        dst_reg_count,
        packed_winv_value,
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {output_cb_id, output_tiles_per_row};
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/"
        "rms_pre_allgather_reader.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/"
        "rms_pre_allgather_writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    const bool use_float32_reduction = fp32_dest_acc_en;  // legacy_reduction = false
    std::vector<uint32_t> compute_args = {
        input_cb_id,
        reduce_scalar_cb_id,
        intermediate_cb_id,
        output_cb_id,
        num_tile_cols,
        dst_reg_count,
        use_float32_reduction,
    };

    const auto* compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/"
        "rmsnorm_pre_allgather.cpp";
    auto compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_args};
    auto compute_kernels_id = CreateKernel(program, compute_kernel_file, core_grid, compute_config);

    const auto cores = corerange_to_cores(core_grid, num_cores, true);

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);

        const uint32_t tile_row_start = std::min(core_id * num_tile_rows_per_core, num_tile_rows);
        const uint32_t tile_row_end = std::min(tile_row_start + num_tile_rows_per_core, num_tile_rows);
        const uint32_t num_tile_rows_to_process = tile_row_end - tile_row_start;

        std::vector<uint32_t> reader_runtime_args = {
            input_addr,
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

    return {
        std::move(program),
        FusedRMSNormPreAllGatherSharedVariables{
            .reader_kernel_id = reader_kernels_id,
            .writer_kernel_id = writer_kernels_id,
            .cores = cores,
        }};
}

void FusedRMSNormPreAllGatherProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FusedRmsnormPreAllGatherParams& /*operation_attributes*/,
    const FusedRmsnormPreAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto input_addr = input_tensor.buffer()->address();
    const auto output_addr = output_tensor.buffer()->address();

    auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);

    for (const auto& core : cores) {
        {
            auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);
            reader_args[0] = input_addr;
        }

        {
            auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
            writer_args[0] = output_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim
