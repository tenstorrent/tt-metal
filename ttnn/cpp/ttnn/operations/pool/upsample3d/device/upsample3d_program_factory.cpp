// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample3d_program_factory.hpp"

#include <math.h>
#include <cstdint>
#include <string>

#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::upsample3d {

tt::tt_metal::operation::ProgramWithCallbacks upsample3d_multi_core_interleaved(
    const Tensor& input,
    Tensor& output,
    const uint32_t scale_factor_d,
    const uint32_t scale_factor_h,
    const uint32_t scale_factor_w) {
    tt::tt_metal::Program program{};

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();

    tt::tt_metal::IDevice* device = output.device();

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    // Calculate proper unit sizes following upsample 2D pattern
    const uint32_t input_unit_size = input.padded_shape()[-1] * input.element_size();  // One row with all channels
    const uint32_t output_unit_size = output.padded_shape()[-1] * output.element_size();
    const uint32_t aligned_input_unit_size = tt::round_up(input_unit_size, tt::tt_metal::hal::get_dram_alignment());

    // For row-major 3D tensor, work unit is one row (stick)
    const uint32_t work_units_to_split = input.physical_volume() / input.padded_shape()[-1];  // N*D*H*W unit split

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Use proper multi-core work distribution like 2D upsample
    const auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_group_1, work_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, work_units_to_split);

    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Create circular buffers with proper sizing
    uint32_t next_cb_index = tt::CBIndex::c_0;
    uint32_t input_cb_required_pages = 1;  // One input row per page
    uint32_t num_pages_in_input_cb = input_cb_required_pages;

    if (work_per_core_group_1 > 1) {
        // Double buffer if processing multiple blocks
        num_pages_in_input_cb *= 2;
    }

    const auto [src0_cb_index, cb_src0] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, aligned_input_unit_size, num_pages_in_input_cb, input_cb_data_format);

    // Use same CB for input and output for row-major
    const uint32_t output_cb_index = src0_cb_index;

    const auto src_buffer = input.buffer();
    const auto dst_buffer = output.buffer();
    const bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // Reader compile time arguments with TensorAccessor
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index, (std::uint32_t)aligned_input_unit_size};
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    const tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample3d/device/kernels/dataflow/reader_upsample3d.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Writer compile time arguments for 3D upsampling with TensorAccessor
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)output_unit_size,
        (std::uint32_t)scale_factor_d,
        (std::uint32_t)scale_factor_h,
        (std::uint32_t)scale_factor_w,
        (std::uint32_t)output_shape[1],  // output_D
        (std::uint32_t)output_shape[2],  // output_H
        (std::uint32_t)output_shape[3],  // output_W
        (std::uint32_t)1,                // block_height for row-major
        (std::uint32_t)1                 // num_units_per_output_stick
    };
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    const tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample3d/device/kernels/dataflow/writer_upsample3d.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Set up runtime arguments for all cores
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

    // Set runtime args per core with proper work distribution
    for (uint32_t i = 0, blocks_processed = 0; i < num_cores; i++) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
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

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_arguments);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_arguments);

        blocks_processed += blocks_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, num_cores, num_cores_y](
                                              const void* operation,
                                              tt::tt_metal::Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        const auto src_buffer = input_tensors.at(0).buffer();
        const auto dst_buffer = output_tensors.at(0).buffer();

        // Update runtime args for all cores
        for (uint32_t i = 0; i < num_cores; i++) {
            const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::upsample3d
