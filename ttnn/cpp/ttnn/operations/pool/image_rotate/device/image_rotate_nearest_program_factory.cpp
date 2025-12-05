// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "image_rotate_device_operation.hpp"

#include <cmath>
#include <cstdint>
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::image_rotate {

using namespace tt;
using namespace tt::tt_metal;

// Constants for nearest interpolation (much simpler than bilinear)
constexpr uint32_t NEAREST_BUFFERING_FACTOR = 2;

// Helper function to decide whether to use split reader for nearest mode
static bool should_use_split_reader_nearest(const Tensor& input_tensor, const Tensor& output_tensor) {
    // Enable split reader for nearest mode to improve performance by distributing work
    // Similar to grid_sample nearest mode
    return true;
}

// Helper to convert float to bfloat16 representation (nearest-specific)
static uint16_t nearest_float_to_bfloat16(float value) {
    return static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16);
}

ImageRotateDeviceOperation::NearestProgramFactory::cached_program_t
ImageRotateDeviceOperation::NearestProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = output;

    tt::tt_metal::Program program{};

    // Data formats
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::tt_metal::IDevice* const device = output_tensor.device();

    // Shape and dimensions (NHWC format)
    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_batch = input_shape[0];
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];
    const uint32_t input_channels = input_shape[3];

    // Calculate rotation parameters
    const float angle_rad = operation_attributes.angle * M_PI / 180.0f;
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);

    // Center point
    // PyTorch uses pixel centers at 0.5, 1.5, ... while we use 0, 1, ...
    // So we subtract 0.5 from PyTorch-style coordinates to match our convention
    // Center point
    // PyTorch uses pixel centers at 0.5, 1.5, ... while we use 0, 1, ...
    // So we subtract 0.5 from PyTorch-style coordinates to match our convention
    float center_x, center_y;
    if (operation_attributes.center.has_value()) {
        center_x = std::get<0>(operation_attributes.center.value()) - 0.5f;
        center_y = std::get<1>(operation_attributes.center.value()) - 0.5f;
    } else {
        center_x = (static_cast<float>(input_width) - 1.0f) / 2.0f;
        center_y = (static_cast<float>(input_height) - 1.0f) / 2.0f;
    }

    // Fill value as bfloat16
    const uint16_t fill_value_bf16 = nearest_float_to_bfloat16(operation_attributes.fill);

    // Decide whether to use split reader
    const bool enable_split_reader = should_use_split_reader_nearest(input_tensor, output_tensor);

    // Work distribution - Total work units = N * H * W (one output pixel per work unit)
    const uint32_t total_output_sticks = input_batch * input_height * input_width;

    // Calculate cores needed
    const auto compute_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);

    const uint32_t num_cores_y = compute_grid_size.y;

    // Get logical cores for setting runtime args
    std::vector<CoreCoord> logical_cores = corerange_to_cores(all_cores, num_cores, true);

    // Calculate stick sizes (aligned to 32 bytes)
    const uint32_t element_size = input_tensor.element_size();
    const uint32_t input_stick_nbytes = input_channels * element_size;
    const uint32_t output_stick_nbytes = input_channels * element_size;

    // CB indices - Much simpler for nearest mode
    uint32_t cb_idx = tt::CBIndex::c_0;

    // CB_0: Output CB for computed output sticks (RISCV_0 reader)
    const uint32_t output_cb_page_size = output_stick_nbytes;
    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++, program, all_cores, output_cb_page_size, NEAREST_BUFFERING_FACTOR, output_cb_data_format);

    // CB_1: Output CB for split reader mode (RISCV_1 reader) - only created if split reader enabled
    uint32_t output_cb_index_1 = 0;
    if (enable_split_reader) {
        const auto [cb_idx_1, cb_handle_1] = tt::tt_metal::create_cb(
            cb_idx++, program, all_cores, output_cb_page_size, NEAREST_BUFFERING_FACTOR, output_cb_data_format);
        output_cb_index_1 = cb_idx_1;
    }

    // Reader/Writer compile-time arguments for nearest mode (RISCV_0 gets CB_0)
    std::vector<uint32_t> reader_writer_compile_time_args = {
        output_cb_index,                // ct_arg[0]: output_cb_index (CB_0 for RISCV_0)
        input_stick_nbytes,             // ct_arg[1]: input_stick_nbytes
        output_stick_nbytes,            // ct_arg[2]: output_stick_nbytes
        input_batch,                    // ct_arg[3]: input_batch
        input_height,                   // ct_arg[4]: input_height
        input_width,                    // ct_arg[5]: input_width
        input_channels,                 // ct_arg[6]: input_channels
        enable_split_reader ? 1U : 0U,  // ct_arg[7]: enable_split_reader
        0U,                             // ct_arg[8]: reader_id
    };

    // Append tensor accessor args for input tensor
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_writer_compile_time_args);
    // Append tensor accessor args for output tensor
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(reader_writer_compile_time_args);

    // Create reader+writer kernel (combined for nearest mode)
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/image_rotate/device/kernels/dataflow/"
        "reader_writer_image_rotate_nearest_interleaved.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_writer_compile_time_args});

    // Create second reader+writer kernel for split reader mode
    tt::tt_metal::KernelHandle reader1_kernel_id = 0;
    if (enable_split_reader) {
        // RISCV_1 gets CB_1
        std::vector<uint32_t> reader1_compile_time_args = {
            output_cb_index_1,              // ct_arg[0]: output_cb_index (CB_1 for RISCV_1)
            input_stick_nbytes,             // ct_arg[1]: input_stick_nbytes
            output_stick_nbytes,            // ct_arg[2]: output_stick_nbytes
            input_batch,                    // ct_arg[3]: input_batch
            input_height,                   // ct_arg[4]: input_height
            input_width,                    // ct_arg[5]: input_width
            input_channels,                 // ct_arg[6]: input_channels
            enable_split_reader ? 1U : 0U,  // ct_arg[7]: enable_split_reader
            1U,                             // ct_arg[8]: reader_id = 1
        };

        // Append tensor accessor args for input tensor
        tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader1_compile_time_args);
        // Append tensor accessor args for output tensor
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(reader1_compile_time_args);

        reader1_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/image_rotate/device/kernels/dataflow/"
            "reader_writer_image_rotate_nearest_interleaved.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = reader1_compile_time_args});
    }

    // Set runtime arguments for each core
    uint32_t sticks_processed = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];
        const uint32_t num_sticks =
            core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

        // Reader+Writer runtime args (combined kernel)
        std::vector<uint32_t> reader_runtime_args = {
            input_tensor.buffer()->address(),       // rt_arg[0]: input_buffer_address
            output_tensor.buffer()->address(),      // rt_arg[1]: output_buffer_address
            num_sticks,                             // rt_arg[2]: num_sticks
            sticks_processed,                       // rt_arg[3]: start_stick_id
            std::bit_cast<uint32_t>(cos_angle),     // rt_arg[4]: cos_angle (as uint32 bits)
            std::bit_cast<uint32_t>(sin_angle),     // rt_arg[5]: sin_angle (as uint32 bits)
            std::bit_cast<uint32_t>(center_x),      // rt_arg[6]: center_x (as uint32 bits)
            std::bit_cast<uint32_t>(center_y),      // rt_arg[7]: center_y (as uint32 bits)
            static_cast<uint32_t>(fill_value_bf16)  // rt_arg[8]: fill_value (bfloat16)
        };

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        if (enable_split_reader) {
            tt::tt_metal::SetRuntimeArgs(program, reader1_kernel_id, core, reader_runtime_args);
        }

        sticks_processed += num_sticks;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = reader_kernel_id,    // Same kernel handles both read and write
         .reader1_kernel_id = reader1_kernel_id,  // Second reader for split mode
         .num_cores = num_cores,
         .num_cores_y = num_cores_y,
         .enable_split_reader = enable_split_reader}};
}

void ImageRotateDeviceOperation::NearestProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& reader1_kernel_id = cached_program.shared_variables.reader1_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;
    auto& enable_split_reader = cached_program.shared_variables.enable_split_reader;

    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = output.buffer();

    // Recalculate rotation parameters
    const float angle_rad = operation_attributes.angle * M_PI / 180.0f;
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);

    // Center point
    const auto& input_shape = tensor_args.input.padded_shape();
    const uint32_t input_width = input_shape[2];
    const uint32_t input_height = input_shape[1];

    // Center point
    // PyTorch uses pixel centers at 0.5, 1.5, ... while we use 0, 1, ...
    // So we subtract 0.5 from PyTorch-style coordinates to match our convention
    float center_x, center_y;
    if (operation_attributes.center.has_value()) {
        center_x = std::get<0>(operation_attributes.center.value()) - 0.5f;
        center_y = std::get<1>(operation_attributes.center.value()) - 0.5f;
    } else {
        center_x = (static_cast<float>(input_width) - 1.0f) / 2.0f;
        center_y = (static_cast<float>(input_height) - 1.0f) / 2.0f;
    }

    const uint16_t fill_value_bf16 = nearest_float_to_bfloat16(operation_attributes.fill);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
            // Update rotation parameters (rt_args 4-8)
            runtime_args[4] = std::bit_cast<uint32_t>(cos_angle);
            runtime_args[5] = std::bit_cast<uint32_t>(sin_angle);
            runtime_args[6] = std::bit_cast<uint32_t>(center_x);
            runtime_args[7] = std::bit_cast<uint32_t>(center_y);
            runtime_args[8] = static_cast<uint32_t>(fill_value_bf16);
        }

        if (enable_split_reader) {
            auto& runtime_args1 = GetRuntimeArgs(program, reader1_kernel_id, core);
            runtime_args1[0] = src_buffer->address();
            runtime_args1[1] = dst_buffer->address();
            // Update rotation parameters (rt_args 4-8)
            runtime_args1[4] = std::bit_cast<uint32_t>(cos_angle);
            runtime_args1[5] = std::bit_cast<uint32_t>(sin_angle);
            runtime_args1[6] = std::bit_cast<uint32_t>(center_x);
            runtime_args1[7] = std::bit_cast<uint32_t>(center_y);
            runtime_args1[8] = static_cast<uint32_t>(fill_value_bf16);
        }
    }
}

}  // namespace ttnn::operations::image_rotate
