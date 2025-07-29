// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <cstdint>
#include <string>

#include "hostdevcommon/kernel_structs.h"
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

#include <tt_stl/reflection.hpp>

namespace ttnn::operations::upsample {
tt::tt_metal::operation::ProgramWithCallbacks upsample_tiled_interleaved(
    const Tensor& input, Tensor& output, const uint32_t scale_factor_h, const uint32_t scale_factor_w) {
    tt::tt_metal::Program program{};

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    const ttnn::Shape& output_shape = output.padded_shape();
    tt::tt_metal::IDevice* const device = output.device();

    const uint32_t input_tensor_width = input.padded_shape()[-1];
    const uint32_t input_tensor_height = input.physical_volume() / input_tensor_width;

    const auto& tile_shape = input.tensor_spec().tile().get_tile_shape();
    const uint32_t tile_height = tile_shape[0];
    const uint32_t tile_width = tile_shape[1];

    const uint32_t num_input_tiles_in_row = input_tensor_width / tile_width;
    const uint32_t num_input_tiles_in_col = input_tensor_height / tile_height;

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_input_tiles_in_col);

    // Input CB

    uint32_t next_cb_index = tt::CBIndex::c_0;
    uint32_t num_pages_in_input_cb = num_input_tiles_in_row;
    if (num_tile_rows_per_core_group_1 != 1) {
        // Double buffer if the core is processing 2+ blocks
        num_pages_in_input_cb *= 2;
    }

    const auto [src0_cb_index, cb_src0] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, input_single_tile_size, num_pages_in_input_cb, input_cb_data_format);

    // Output CB

    uint32_t num_pages_in_output_cb = num_input_tiles_in_row;
    if (num_tile_rows_per_core_group_1 != 1) {
        num_pages_in_output_cb *= 2;
    }

    const auto [output_cb_index, cb_output] = create_cb(
        next_cb_index++, program, all_cores, output_single_tile_size, num_pages_in_output_cb, output_cb_data_format);

    const tt::tt_metal::Buffer* src_buffer = input.buffer();
    const tt::tt_metal::Buffer* dst_buffer = output.buffer();
    const bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    /*
    The data layout is mapped in DRAM as follows:
        number of channel = stick size
        NHW is number of sticks
    */

    // Reader compile time arguments

    std::vector<uint32_t> reader_compile_time_args;
    const bool src_tile_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(input_single_tile_size);
    const uint32_t src_log2_tile_size = src_tile_size_is_power_of_two ? (std::uint32_t)log2(input_single_tile_size) : 0;

    reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)input_single_tile_size,
        (std::uint32_t)src_tile_size_is_power_of_two,
        (std::uint32_t)src_log2_tile_size};

    const tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/"
        "reader_upsample_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    const uint32_t output_stick_size = output.padded_shape()[-1] * output.element_size();

    const bool dst_stick_size_is_pow2 = tt::tt_metal::is_power_of_two_at_least_32(output_stick_size);
    const uint32_t dst_log_base_2_of_page_size = dst_stick_size_is_pow2 ? (std::uint32_t)log2(output_stick_size) : 0;

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)dst_is_dram,
        (std::uint32_t)output_stick_size,
        (std::uint32_t)dst_stick_size_is_pow2,
        (std::uint32_t)dst_log_base_2_of_page_size,
        (std::uint32_t)scale_factor_h,
        (std::uint32_t)scale_factor_w,
        (std::uint32_t)output_shape[1],
        (std::uint32_t)output_shape[2],
        (std::uint32_t)tile_height,
        (std::uint32_t)num_input_tiles_in_row,
    };

    const tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Compute kernel

    const std::vector<uint32_t> compute_compile_time_args = {
        (uint32_t)num_input_tiles_in_row, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};

    const tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/untilize.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    std::vector<uint32_t> reader_rt_arguments{
        src_buffer->address(),
        0,  // set in loop, num of sticks on core
        0   // set in loop, start_id of stick in core
    };

    std::vector<uint32_t> compute_rt_arguments{
        0  // set in loop, number of blocks per core
    };

    std::vector<uint32_t> writer_rt_arguments{
        dst_buffer->address(),
        0,  // set in loop, num of blocks on core
        0   // set in loop, start_id of first input stick
    };

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_of_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_of_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_of_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        reader_rt_arguments[1] = num_of_tile_rows_per_core * num_input_tiles_in_row;
        reader_rt_arguments[2] = num_tiles_read * num_input_tiles_in_row;

        writer_rt_arguments[1] = num_of_tile_rows_per_core;
        writer_rt_arguments[2] = num_tiles_read;

        compute_rt_arguments[0] = num_of_tile_rows_per_core;

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_arguments);

        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_arguments);

        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_rt_arguments);

        num_tiles_read += num_of_tile_rows_per_core;
    }

    auto override_runtime_args_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, compute_kernel_id, num_cores, num_cores_y](
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
