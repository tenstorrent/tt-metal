// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_block_interleaved_program_factory.hpp"
#include "tilize_with_val_padding_single_core_program_factory.hpp"

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding_common.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

// get_packed_value function is defined in single_core_program_factory.cpp

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_block_interleaved(
    const Tensor& a, Tensor& output, const ttnn::PadValue pad_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();

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
            ttnn::split_blocks_for_tilize_wh(grid_size, num_blocks, num_tiles_per_row, num_tiles_per_col);

    uint32_t total_tiles_per_row = full_cores_per_row * single_block_size + has_cliff_row * single_block_size_cliff_row;

    uint32_t unpadded_row_size_bytes = a.padded_shape()[-1] * a.element_size();     // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output.padded_shape()[-1] * a.element_size();  // Assuming bfloat16 dataformat

    const uint32_t onetile = 1;
    if (core_range.size() > 0) {
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
        auto [src3_cb_index, cb_src3] = create_cb(
            tt::CBIndex::c_0,
            program,
            cliff_col_core_range,
            input_single_tile_size,
            single_block_size,
            input_cb_data_format);

        auto [output3_cb_index, cb_output3] = create_cb(
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

    uint32_t src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_row_size_bytes;
    uint32_t stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(stick_size) : 0;

    uint32_t packed_pad_value = get_packed_value(a, pad_value);
    // log2(TILE_WIDTH * data_format_size_in_bytes)
    uint32_t shift_bits = (a.dtype() == DataType::BFLOAT16) ? 6 : 7;

    uint32_t num_tiles_2d = output.padded_shape()[-1] * output.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    uint32_t tile_width = output.tensor_spec().tile().get_width();
    uint32_t tile_height = output.tensor_spec().tile().get_height();

    uint32_t total_num_rows = a.logical_shape()[-2];

    if (output.padded_shape()[-2] > tt::round_up(total_num_rows, tile_height)) {
        total_num_rows = output.padded_shape()[-2];
    }

    std::map<std::string, std::string> reader_defines = {
        {"STICK_SIZE_IS_POW2", std::to_string((uint32_t)(stick_size_is_power_of_two))}};

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "reader_unary_pad_multicore_both_dims.cpp",
        all_cores,
        ReaderDataMovementConfig(
            {src0_is_dram, log2_stick_size, total_num_rows, third_dim, tile_height, a.element_size()}, reader_defines));

    // writer
    uint32_t out_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp",
        all_cores,
        WriterDataMovementConfig({tt::CBIndex::c_16, out_is_dram, num_tiles_2d, third_dim, total_tiles_per_row}));

    // compute

    if (core_range.size() > 0) {
        auto tilize_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp",
            core_range,
            ComputeConfig{.compile_args = {single_block_size, single_block_size, third_dim}});
    }
    if (has_cliff_col && has_cliff_row) {
        auto tilize_col_row_cliff_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp",
            cliff_col_row_core_range,
            ComputeConfig{.compile_args = {single_block_size_cliff_col, single_block_size_cliff_row, third_dim}});
    }
    if (has_cliff_row) {
        auto tilize_row_cliff_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp",
            cliff_row_core_range,
            ComputeConfig{.compile_args = {single_block_size, single_block_size_cliff_row, third_dim}});
    }

    if (has_cliff_col) {
        auto tilize_col_cliff_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp",
            cliff_col_core_range,
            ComputeConfig{.compile_args = {single_block_size_cliff_col, single_block_size, third_dim}});
    }

    // RUNTIME ARGS
    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
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
        uint32_t size_per_row_per_block = nblocks_per_core * TILE_WIDTH * a.element_size();

        //  reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            src0_buffer->address(),
            unpadded_row_size_bytes,
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

        uint32_t end_column_id = start_column_id + single_block_size_row_arg * TILE_WIDTH * a.element_size();
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

    auto override_runtime_args_callback =
        [reader_kernel_id = unary_reader_kernel_id, writer_kernel_id = unary_writer_kernel_id, cores = cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors.at(0).buffer();
            auto dst_buffer = output_tensors.at(0).buffer();

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = src_buffer->address();
                }
                {
                    auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = dst_buffer->address();
                }
            }
        };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::detail
