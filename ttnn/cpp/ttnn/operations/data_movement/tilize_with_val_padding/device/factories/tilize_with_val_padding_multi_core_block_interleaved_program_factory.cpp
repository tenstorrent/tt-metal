// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_block_interleaved_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

TilizeWithValPaddingMultiCoreBlockInterleavedFactory::cached_program_t
TilizeWithValPaddingMultiCoreBlockInterleavedFactory::create(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor, const Tensor& output_tensor) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const Tensor& a = input_tensor;
    const Tensor& output = output_tensor;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t num_tiles_per_col = output.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = output.padded_shape()[-1] / TILE_WIDTH;

    uint32_t num_blocks = (output.padded_shape()[-1] * output.padded_shape()[-2]) / (TILE_HEIGHT * TILE_WIDTH);

    auto
        [ncores,
         all_cores,
         core_range,
         cliff_row_core_range,
         cliff_col_core_range,
         cliff_col_row_core_range,
         nblocks_per_core,
         single_block_size,
         single_block_size_cliff_row,
         single_block_size_cliff_col,
         has_cliff_row,
         has_cliff_col,
         full_cores_per_row,
         full_cores_per_col] =
            ttnn::split_blocks_for_tilize_wh(available_grid, num_blocks, num_tiles_per_row, num_tiles_per_col);

    uint32_t total_tiles_per_row =
        (full_cores_per_row * single_block_size) + (has_cliff_row * single_block_size_cliff_row);

    uint32_t unpadded_row_size_bytes = a.padded_shape()[-1] * a.element_size();     // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output.padded_shape()[-1] * a.element_size();  // Assuming bfloat16 dataformat

    if (!core_range.empty()) {
        create_cb(
            tt::CBIndex::c_0, program, core_range, input_single_tile_size, single_block_size, input_cb_data_format);

        create_cb(
            tt::CBIndex::c_16, program, core_range, output_single_tile_size, single_block_size, output_cb_data_format);
    }
    if (has_cliff_col && has_cliff_row) {
        create_cb(
            tt::CBIndex::c_0,
            program,
            cliff_col_row_core_range,
            input_single_tile_size,
            single_block_size_cliff_row,
            input_cb_data_format);

        create_cb(
            tt::CBIndex::c_16,
            program,
            cliff_col_row_core_range,
            output_single_tile_size,
            single_block_size_cliff_row,
            output_cb_data_format);
    }

    if (has_cliff_row) {
        create_cb(
            tt::CBIndex::c_0,
            program,
            cliff_row_core_range,
            input_single_tile_size,
            single_block_size_cliff_row,
            input_cb_data_format);

        create_cb(
            tt::CBIndex::c_16,
            program,
            cliff_row_core_range,
            output_single_tile_size,
            single_block_size_cliff_row,
            output_cb_data_format);
    }

    if (has_cliff_col) {
        create_cb(
            tt::CBIndex::c_0,
            program,
            cliff_col_core_range,
            input_single_tile_size,
            single_block_size,
            input_cb_data_format);

        create_cb(
            tt::CBIndex::c_16,
            program,
            cliff_col_core_range,
            output_single_tile_size,
            single_block_size,
            output_cb_data_format);
    }

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // reader
    uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);
    // log2(TILE_WIDTH * data_format_size_in_bytes)

    uint32_t num_tiles_2d = output.padded_shape()[-1] * output.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    uint32_t tile_height = output.tensor_spec().tile().get_height();

    uint32_t total_num_rows = a.logical_shape()[-2];

    if (output.padded_shape()[-2] > tt::round_up(total_num_rows, tile_height)) {
        total_num_rows = output.padded_shape()[-2];
    }

    std::vector<uint32_t> reader_compile_time_args = {
        total_num_rows, third_dim, tile_height, a.element_size(), unpadded_row_size_bytes};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "reader_unary_pad_multicore_both_dims.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    // writer

    std::vector<uint32_t> writer_compile_time_args = {tt::CBIndex::c_16, num_tiles_2d, third_dim, total_tiles_per_row};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    // compute

    if (!core_range.empty()) {
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp",
            core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_llk_acc, .compile_args = {single_block_size, single_block_size, third_dim}});
    }
    if (has_cliff_col && has_cliff_row) {
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp",
            cliff_col_row_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
                .compile_args = {single_block_size_cliff_col, single_block_size_cliff_row, third_dim}});
    }
    if (has_cliff_row) {
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp",
            cliff_row_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
                .compile_args = {single_block_size, single_block_size_cliff_row, third_dim}});
    }

    if (has_cliff_col) {
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp",
            cliff_col_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
                .compile_args = {single_block_size_cliff_col, single_block_size, third_dim}});
    }

    // RUNTIME ARGS
    const auto cores = corerange_to_cores(available_grid);
    uint32_t start_row_id = 0;
    uint32_t start_column_id = 0;
    uint32_t tile_start_id = 0;
    uint32_t single_block_size_row_arg;
    uint32_t single_block_size_col_arg;

    uint32_t total_row_cores = full_cores_per_row;
    if (has_cliff_row) {
        total_row_cores++;
    }
    uint32_t cores_col_count = 1;
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        if (has_cliff_col && has_cliff_row && i == ncores - 1) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size_cliff_col;

        } else if (has_cliff_row && i != 0 && ((i + 1) % (full_cores_per_row + 1)) == 0) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size;

        } else if (i < total_row_cores * full_cores_per_col) {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size;

        } else {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size_cliff_col;
        }

        //  reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            src0_buffer->address(),
            packed_pad_value,
            TILE_WIDTH * a.element_size() * single_block_size_row_arg,
            start_row_id,
            start_column_id,
            single_block_size_row_arg,
            single_block_size_col_arg,
        };

        // writer runtime args
        const std::array writer_rt_args = {
            dst_buffer->address(), tile_start_id, single_block_size_row_arg, single_block_size_col_arg};

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        uint32_t end_column_id = start_column_id + (single_block_size_row_arg * TILE_WIDTH * a.element_size());
        start_column_id = end_column_id % padded_row_size_bytes;
        if (end_column_id % padded_row_size_bytes == 0 && end_column_id != 0) {
            start_row_id += single_block_size_col_arg * TILE_HEIGHT;
        }

        if (start_column_id == 0) {
            tile_start_id = cores_col_count * single_block_size_col_arg * total_tiles_per_row;
            cores_col_count++;
        } else {
            tile_start_id += single_block_size_row_arg;
        }
    }

    shared_variables_t shared_variables{
        .reader_kernel_id = unary_reader_kernel_id,
        .writer_kernel_id = unary_writer_kernel_id,
        .cores = cores,
        .ncores = ncores};
    return cached_program_t(std::move(program), std::move(shared_variables));
}

void TilizeWithValPaddingMultiCoreBlockInterleavedFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const Tensor& input_tensor,
    const Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    const auto& ncores = shared_variables.ncores;
    const auto& cores = shared_variables.cores;
    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    auto& reader_runtime_args_by_core = GetRuntimeArgs(program, shared_variables.reader_kernel_id);
    auto& writer_runtime_args_by_core = GetRuntimeArgs(program, shared_variables.writer_kernel_id);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        {
            auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = src_buffer->address();
        }
        {
            auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
