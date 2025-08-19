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

tt::tt_metal::operation::ProgramWithCallbacks upsample3d_multi_core_height_sharded(
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

    // Get input and output shard specs for height sharded tensors
    const auto input_shard_spec = input.shard_spec().value();
    const auto output_shard_spec = output.shard_spec().value();

    TT_FATAL(
        input_shard_spec.num_cores() == output_shard_spec.num_cores(),
        "Input and output shard specs must have the same number of cores for height sharded upsample3d");

    const uint32_t stick_nbytes = input.padded_shape()[-1] * input.element_size();
    const uint32_t aligned_stick_nbytes = tt::round_up(stick_nbytes, tt::tt_metal::hal::get_dram_alignment());

    // Get all cores used for sharding
    auto all_cores = input_shard_spec.grid;
    const uint32_t num_cores = input_shard_spec.num_cores();

    // No circular buffers needed for direct copy approach
    const uint32_t input_cb_index = 0;  // Dummy values for compile-time args
    const uint32_t output_cb_index = 1;

    const auto input_buffer = input.buffer();
    const auto output_buffer = output.buffer();

    // Single kernel compile time arguments (following upsample 2D pattern)
    std::vector<uint32_t> kernel_compile_time_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)true,  // is_reader flag for NCRISC (will create two instances)
        (std::uint32_t)aligned_stick_nbytes,
        (std::uint32_t)scale_factor_d,
        (std::uint32_t)scale_factor_h,
        (std::uint32_t)scale_factor_w,
        (std::uint32_t)output_shape[1],  // output_d
        (std::uint32_t)output_shape[2],  // output_h
        (std::uint32_t)output_shape[3],  // output_w
    };

    // Add input TensorAccessor args
    tt::tt_metal::TensorAccessorArgs(*input_buffer).append_to(kernel_compile_time_args);
    // Add output TensorAccessor args
    tt::tt_metal::TensorAccessorArgs(*output_buffer).append_to(kernel_compile_time_args);

    // Create separate NCRISC and BRISC kernels
    std::vector<uint32_t> reader_compile_time_args = kernel_compile_time_args;
    reader_compile_time_args[2] = true;  // is_reader = true for NCRISC

    std::vector<uint32_t> writer_compile_time_args = kernel_compile_time_args;
    writer_compile_time_args[2] = false;  // is_reader = false for BRISC

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample3d/device/kernels/dataflow/writer_upsample3d_height_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample3d/device/kernels/dataflow/writer_upsample3d_height_sharded.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Calculate pages per core for height sharded tensors
    const auto total_output_pages = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];  // N*D*H*W
    const uint32_t pages_per_core = (total_output_pages + num_cores - 1) / num_cores;

    // Set runtime arguments for all cores
    std::vector<uint32_t> runtime_arguments{
        input_buffer->address(),
        output_buffer->address(),
        0,  // num_output_pages (will be set per core)
        0   // start_output_page_id (will be set per core)
    };

    // Work distribution: each core processes its assigned output pages
    auto logical_cores = corerange_to_cores(
        output_shard_spec.grid,
        output_shard_spec.num_cores(),
        output_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = logical_cores[i];

        // Calculate pages for this core
        uint32_t start_page_id = i * pages_per_core;
        uint32_t end_page_id = std::min((i + 1) * pages_per_core, total_output_pages);
        uint32_t num_pages_this_core = end_page_id - start_page_id;

        runtime_arguments[2] = num_pages_this_core;  // num_output_pages
        runtime_arguments[3] = start_page_id;        // start_output_page_id

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, runtime_arguments);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, runtime_arguments);
    }

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, num_cores](
                                              const void* operation,
                                              tt::tt_metal::Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        const auto input_buffer = input_tensors.at(0).buffer();
        const auto output_buffer = output_tensors.at(0).buffer();

        const auto output_shard_spec = output_tensors.at(0).shard_spec().value();
        auto logical_cores = corerange_to_cores(
            output_shard_spec.grid,
            output_shard_spec.num_cores(),
            output_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

        // Update runtime args for all cores
        for (uint32_t i = 0; i < num_cores; ++i) {
            const auto& core = logical_cores[i];
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_buffer->address();
                runtime_args[1] = output_buffer->address();
            }
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = input_buffer->address();
                runtime_args[1] = output_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::upsample3d
