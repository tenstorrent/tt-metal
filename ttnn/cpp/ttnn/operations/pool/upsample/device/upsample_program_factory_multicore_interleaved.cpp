// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <cstdint>
#include <string>

#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/work_split.hpp"
#include "upsample_op.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::upsample {

tt::tt_metal::operation::ProgramWithCallbacks upsample_multi_core_interleaved(
    const Tensor& input, Tensor& output, const uint32_t scale_factor_h, const uint32_t scale_factor_w) {
    tt::tt_metal::Program program{};

    const bool is_tiled_layout = (input.layout() == tt::tt_metal::Layout::TILE);

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    const auto& output_shape = output.padded_shape();
    tt::tt_metal::IDevice* const device = output.device();

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Declare variables that will be set based on layout
    uint32_t input_unit_size;
    uint32_t output_unit_size;
    uint32_t input_cb_required_pages;
    uint32_t work_units_to_split;
    uint32_t aligned_input_unit_size;  // Size used for CB creation

    if (is_tiled_layout) {
        // Tiled layout specific calculations
        input_unit_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
        output_unit_size = tt::tt_metal::detail::TileSize(output_cb_data_format);
        aligned_input_unit_size = input_unit_size;

        const uint32_t input_tensor_width = input.padded_shape()[-1];
        const uint32_t input_tensor_height = input.physical_volume() / input_tensor_width;

        const auto& tile_shape = input.tensor_spec().tile().get_tile_shape();
        const uint32_t tile_height = tile_shape[0];
        const uint32_t tile_width = tile_shape[1];

        const uint32_t num_input_tiles_in_row = input_tensor_width / tile_width;
        const uint32_t num_input_tiles_in_col = input_tensor_height / tile_height;

        /*
        For tiled layout, a unit of work (input wise) is a row of tiles
        */

        input_cb_required_pages = num_input_tiles_in_row;  // for CB sizing
        work_units_to_split = num_input_tiles_in_col;  // for work splitting
    } else {
        // Row-major layout specific calculations
        input_unit_size = input.padded_shape()[-1] * input.element_size();
        output_unit_size = output.padded_shape()[-1] * output.element_size();
        aligned_input_unit_size = tt::round_up(input_unit_size, tt::tt_metal::hal::get_dram_alignment());

        /*
        For Row-major layout, a unit of work is one row (stick) of the input tensor
        */

        input_cb_required_pages = 1;                                               // One input unit is required in CB
        work_units_to_split = input.physical_volume() / input.padded_shape()[-1];  // N*H*W unit split
    }

    const auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_group_1, work_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, work_units_to_split);

    // Create circular buffers
    uint32_t next_cb_index = tt::CBIndex::c_0;
    uint32_t num_pages_in_input_cb;

    num_pages_in_input_cb = input_cb_required_pages;
    if (work_per_core_group_1 != 1) {
        // Double buffer if the core is processing 2+ blocks
        num_pages_in_input_cb *= 2;
    }

    const auto [src0_cb_index, cb_src0] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, aligned_input_unit_size, num_pages_in_input_cb, input_cb_data_format);

    uint32_t output_cb_index = 0;
    if (is_tiled_layout) {
        // Separate output CB for tiled
        uint32_t num_pages_in_output_cb = num_pages_in_input_cb;
        const auto [out_cb_index, cb_output] = create_cb(
            next_cb_index++, program, all_cores, output_unit_size, num_pages_in_output_cb, output_cb_data_format);
        output_cb_index = out_cb_index;
    } else {
        // Same CB for input and output for row-major
        output_cb_index = src0_cb_index;
    }

    const auto src_buffer = input.buffer();
    const auto dst_buffer = output.buffer();
    const bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // Reader compile time arguments
    const bool src_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(aligned_input_unit_size);
    const uint32_t src_log2_size = src_size_is_power_of_two ? (std::uint32_t)log2(aligned_input_unit_size) : 0;

    const std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)aligned_input_unit_size,
        (std::uint32_t)src_size_is_power_of_two,
        (std::uint32_t)src_log2_size};

    const tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/"
        "reader_upsample_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Writer compile time arguments

    const int32_t writer_unit_size = output.padded_shape()[-1] * output.element_size();
    const bool dst_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(writer_unit_size);
    const uint32_t dst_log2_size = dst_size_is_power_of_two ? (std::uint32_t)log2(writer_unit_size) : 0;

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)dst_is_dram,
        (std::uint32_t)writer_unit_size,
        (std::uint32_t)dst_size_is_power_of_two,
        (std::uint32_t)dst_log2_size,
        (std::uint32_t)scale_factor_h,
        (std::uint32_t)scale_factor_w,
        (std::uint32_t)output_shape[1],
        (std::uint32_t)output_shape[2]};

    if (is_tiled_layout) {
        const auto& tile_shape = input.tensor_spec().tile().get_tile_shape();
        const uint32_t tile_height = tile_shape[0];
        const uint32_t num_input_tiles_in_row = input.padded_shape()[-1] / tile_shape[1];

        writer_compile_time_args.push_back(
            (std::uint32_t)tile_height);  // tile_height rows need to be processed at a time
        writer_compile_time_args.push_back(
            (std::uint32_t)
                num_input_tiles_in_row);  // whole row of tiles needs to be processed to get valid output sticks
    } else {
        const uint32_t block_height = 1;                // since input is row major, blocks are just one row tall
        const uint32_t num_units_per_output_stick = 1;  // 1 page in out_cb is needed to get a valid output stick

        writer_compile_time_args.push_back((std::uint32_t)block_height);
        writer_compile_time_args.push_back(num_units_per_output_stick);
    }

    const std::map<std::string, std::string> kernel_defines;
    const tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    // Compute kernel (only for tiled layout)
    tt::tt_metal::KernelHandle compute_kernel_group1_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_group2_id = 0;
    if (is_tiled_layout) {
        const uint32_t num_input_tiles_in_row =
            input.padded_shape()[-1] / input.tensor_spec().tile().get_tile_shape()[1];

        // Create compute kernel for core group 1 if it has cores
        if (core_group_1.num_cores() > 0) {
            const std::vector<uint32_t> compute_compile_time_args_group1 = {
                (uint32_t)work_per_core_group_1,   // per_core_block_cnt (compile-time)
                (uint32_t)num_input_tiles_in_row,  // per_block_ntiles
                (uint32_t)src0_cb_index,           // src_cb_id
                (uint32_t)output_cb_index          // out_cb_id
            };

            compute_kernel_group1_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp",
                core_group_1,
                tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args_group1});
        }

        // Create compute kernel for core group 2 if it has cores
        if (core_group_2.num_cores() > 0) {
            const std::vector<uint32_t> compute_compile_time_args_group2 = {
                (uint32_t)work_per_core_group_2,   // per_core_block_cnt (compile-time)
                (uint32_t)num_input_tiles_in_row,  // per_block_ntiles
                (uint32_t)src0_cb_index,           // src_cb_id
                (uint32_t)output_cb_index          // out_cb_id
            };

            compute_kernel_group2_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp",
                core_group_2,
                tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args_group2});
        }
    }

    // Set up runtime arguments
    std::vector<uint32_t> reader_rt_arguments{
        src_buffer->address(),
        0,  // set in loop, num of units on core
        0   // set in loop, start_id of unit in core
    };

    std::vector<uint32_t> writer_rt_arguments{
        dst_buffer->address(),
        0,  // set in loop, num of units on core
        0   // set in loop, start_id of unit on core
    };

    /*
    For tiled input, a block refers to a row of input tiles
    For row-major input, a block refers to a single input row (stick)
    */

    for (uint32_t i = 0, blocks_processed = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            blocks_per_core = work_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            blocks_per_core = work_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        reader_rt_arguments[1] = blocks_per_core * input_cb_required_pages;  // reader goes page by page
        reader_rt_arguments[2] = blocks_processed * input_cb_required_pages;

        writer_rt_arguments[1] = blocks_per_core;
        writer_rt_arguments[2] = blocks_processed;

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_arguments);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_arguments);

        blocks_processed += blocks_per_core;
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id, num_cores, num_cores_y](
                                              const void* operation,
                                              tt::tt_metal::Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        const auto src_buffer = input_tensors.at(0).buffer();
        const auto dst_buffer = output_tensors.at(0).buffer();

        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::upsample
