// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <cstdint>
#include <string>

#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/work_split.hpp"
#include "upsample3D_program_factory.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::upsample {

tt::tt_metal::operation::ProgramWithCallbacks upsample3d_multi_core_interleaved(
    const Tensor& input,
    Tensor& output,
    const uint32_t scale_factor_d,
    const uint32_t scale_factor_h,
    const uint32_t scale_factor_w) {
    tt::tt_metal::Program program{};

    // Validate input tensor
    TT_FATAL(input.get_shape().rank() == 5, "Input tensor must be 5D (N, D, H, W, C)");
    TT_FATAL(input.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Only row-major layout is supported for 3D upsample");
    TT_FATAL(
        input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "Only interleaved memory layout is supported");

    // Get data formats
    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    // Get tensor shapes
    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();

    // Extract dimensions
    const uint32_t N = input_shape[0];  // Batch
    const uint32_t D = input_shape[1];  // Depth
    const uint32_t H = input_shape[2];  // Height
    const uint32_t W = input_shape[3];  // Width
    const uint32_t C = input_shape[4];  // Channels

    // Verify output shape is correct
    TT_FATAL(output_shape[0] == N, "Output batch size mismatch");
    TT_FATAL(output_shape[1] == D * scale_factor_d, "Output depth mismatch");
    TT_FATAL(output_shape[2] == H * scale_factor_h, "Output height mismatch");
    TT_FATAL(output_shape[3] == W * scale_factor_w, "Output width mismatch");
    TT_FATAL(output_shape[4] == C, "Output channels mismatch");

    // Get device and compute grid
    tt::tt_metal::IDevice* const device = output.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Calculate work units
    // Each work unit is one stick (row) of C elements
    const uint32_t input_unit_size = C * input.element_size();  // Size of one stick in bytes
    const uint32_t output_unit_size = C * output.element_size();
    const uint32_t aligned_input_unit_size = tt::round_up(input_unit_size, tt::tt_metal::hal::get_dram_alignment());

    // Total number of sticks in the flattened input tensor
    const uint32_t work_units_to_split = N * D * H * W;

    // Distribute work across cores
    const auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_group_1, work_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, work_units_to_split);

    // Create circular buffers
    uint32_t next_cb_index = tt::CBIndex::c_0;

    // Determine CB size (double buffer if processing multiple sticks)
    uint32_t num_pages_in_input_cb = 1;  // One stick at a time
    if (work_per_core_group_1 > 1) {
        num_pages_in_input_cb *= 2;  // Double buffer
    }

    // Create input CB
    const auto [src0_cb_index, cb_src0] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, aligned_input_unit_size, num_pages_in_input_cb, input_cb_data_format);

    // For row-major layout, we reuse the same CB for output
    const uint32_t output_cb_index = src0_cb_index;

    // Get buffer addresses
    const auto src_buffer = input.buffer();
    const auto dst_buffer = output.buffer();

    // Reader kernel compile-time arguments
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)aligned_input_unit_size,
    };
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args);

    // Create reader kernel (reuse existing 2D reader - it works for sticks)
    const tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/"
        "reader_upsample_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Writer kernel compile-time arguments
    const uint32_t writer_unit_size = C * output.element_size();

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)writer_unit_size,
        (std::uint32_t)scale_factor_d,
        (std::uint32_t)scale_factor_h,
        (std::uint32_t)scale_factor_w,
        (std::uint32_t)output_shape[1],  // Output D
        (std::uint32_t)output_shape[2],  // Output H
        (std::uint32_t)output_shape[3],  // Output W
        (std::uint32_t)D,                // Input D
        (std::uint32_t)H,                // Input H
        (std::uint32_t)W,                // Input W
    };

    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);

    // Create writer kernel
    // NOTE: This assumes writer_upsample3d_interleaved.cpp exists
    // You will need to create this kernel file
    const std::map<std::string, std::string> kernel_defines;
    const tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample3d_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    // Set up runtime arguments
    std::vector<uint32_t> reader_rt_arguments{
        src_buffer->address(),
        0,  // num_sticks - set in loop
        0   // start_stick_id - set in loop
    };

    std::vector<uint32_t> writer_rt_arguments{
        dst_buffer->address(),
        0,  // num_sticks - set in loop
        0   // start_stick_id - set in loop
    };

    // Distribute work to cores and set runtime arguments
    for (uint32_t i = 0, sticks_processed = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t sticks_per_core = 0;

        if (core_group_1.contains(core)) {
            sticks_per_core = work_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            sticks_per_core = work_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        reader_rt_arguments[1] = sticks_per_core;   // Number of sticks to read
        reader_rt_arguments[2] = sticks_processed;  // Starting stick ID

        writer_rt_arguments[1] = sticks_per_core;   // Number of sticks to write
        writer_rt_arguments[2] = sticks_processed;  // Starting stick ID

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_arguments);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_arguments);

        sticks_processed += sticks_per_core;
    }

    // Create override callback to update buffer addresses when tensors move
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
