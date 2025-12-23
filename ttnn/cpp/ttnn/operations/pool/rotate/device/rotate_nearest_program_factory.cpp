// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/pool/rotate/device/rotate_device_operation.hpp>
#include <ttnn/operations/pool/rotate/device/kernels/fixed_point_q16.h>
#include <ttnn/operations/pool/pool_utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <ttnn/operations/cb_utils.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::rotate {

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t NEAREST_BUFFERING_FACTOR = 2;
constexpr uint32_t NUM_TILES_DEST = 8;
// Maximum number of sticks to process per batch for optimal NOC utilization
constexpr uint32_t MAX_BATCH_SIZE = 5;

// Helper to convert float to bfloat16 representation using tie-to-even rounding (matches PyTorch)
static uint16_t nearest_float_to_bfloat16(float value) {
    bfloat16 bf16_value(value);
    return std::bit_cast<uint16_t>(bf16_value);
}

RotateDeviceOperation::NearestProgramFactory::cached_program_t RotateDeviceOperation::NearestProgramFactory::create(
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

    // Work distribution - Total work units = N * H * W (one output pixel per work unit)
    const uint32_t total_output_sticks = input_batch * input_height * input_width;

    // Calculate cores needed
    const auto compute_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);

    const uint32_t num_cores_y = compute_grid_size.y;

    // Get logical cores for setting runtime args
    std::vector<CoreCoord> logical_cores = corerange_to_cores(all_cores, num_cores, true);

    // Calculate stick sizes (aligned based on buffer type for efficient reads)
    const uint32_t element_size = input_tensor.element_size();
    const uint32_t input_stick_nbytes = input_channels * element_size;
    const uint32_t aligned_input_stick_nbytes = pool::get_aligned_stick_size(input_shape, input_tensor);
    const uint32_t aligned_output_stick_nbytes = pool::get_aligned_stick_size(input_shape, output_tensor);

    // Calculate max CB pages based on available L1 memory
    // Due to nothing else taking L1 in this data movement op, let's assume
    // that it is okay to use 16KB of L1 for this CB allocation and calculate accordingly
    const uint32_t available_l1 = NUM_TILES_DEST * tt::constants::TILE_HW * element_size;
    const uint32_t l1_for_cb = available_l1 / NEAREST_BUFFERING_FACTOR;
    const uint32_t max_cb_pages_from_l1 = l1_for_cb / aligned_input_stick_nbytes;

    // Determine actual number of CB pages: min of (max work per core, L1 capacity)
    const uint32_t max_sticks_per_core = std::max(num_sticks_per_core_group_1, num_sticks_per_core_group_2);
    uint32_t num_cb_pages = std::min(max_sticks_per_core, max_cb_pages_from_l1);

    // CB_0: Output CB for communication between reader and writer
    const uint32_t output_cb_page_size = aligned_input_stick_nbytes;
    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        tt::CBIndex::c_0,
        program,
        all_cores,
        output_cb_page_size,
        num_cb_pages * NEAREST_BUFFERING_FACTOR,
        output_cb_data_format);

    // Check if fill value is zero (can use zero_out_page instead of fill CB)
    const bool fill_is_zero = (fill_value_bf16 == 0);

    // CB_1: Fill CB - single page to hold pre-filled stick for L1-to-L1 copy
    uint32_t fill_cb_index = 0;
    auto [cb_index, cb_handle] =
        tt::tt_metal::create_cb(tt::CBIndex::c_1, program, all_cores, output_cb_page_size, 1, output_cb_data_format);
    fill_cb_index = cb_index;
    // Batch size is limited by available CB pages or MAX_BATCH_SIZE, whichever is smaller
    const uint32_t batch_size = num_cb_pages < MAX_BATCH_SIZE ? num_cb_pages : MAX_BATCH_SIZE;

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_time_args = {
        output_cb_index,                      // ct_arg[0]: output_cb_index
        aligned_input_stick_nbytes,           // ct_arg[1]: aligned_input_stick_nbytes (for DRAM reads)
        input_batch,                          // ct_arg[2]: input_batch
        input_height,                         // ct_arg[3]: input_height
        input_width,                          // ct_arg[4]: input_width
        input_channels,                       // ct_arg[5]: input_channels
        num_cb_pages,                         // ct_arg[6]: num_cb_pages
        fill_cb_index,                        // ct_arg[7]: fill_cb_index (0 if fill_is_zero)
        input_stick_nbytes,                   // ct_arg[8]: input_stick_nbytes (unaligned, for fill)
        static_cast<uint32_t>(fill_is_zero),  // ct_arg[9]: fill_is_zero
        batch_size                            // ct_arg[10]: batch_size
    };

    // Append tensor accessor args for input tensor (starts at ct_arg[11])
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    // Writer compile-time arguments (RISCV_1)
    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_index,              // ct_arg[0]: output_cb_index
        aligned_output_stick_nbytes,  // ct_arg[1]: aligned_output_stick_nbytes
        num_cb_pages,                 // ct_arg[2]: num_cb_pages
        batch_size,                   // ct_arg[3]: batch_size
    };

    // Append tensor accessor args for output tensor (starts at ct_arg[4])
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

    // Create reader kernel
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/rotate/device/kernels/dataflow/"
        "reader_rotate_nearest_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Create writer kernel
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/rotate/device/kernels/dataflow/"
        "writer_rotate_nearest_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Set runtime arguments for each core
    uint32_t sticks_processed = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];
        const uint32_t num_sticks =
            core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

        // Reader runtime args
        std::vector<uint32_t> reader_runtime_args = {
            input_tensor.buffer()->address(),                // rt_arg[0]: input_buffer_address
            num_sticks,                                      // rt_arg[1]: num_sticks
            sticks_processed,                                // rt_arg[2]: start_stick_id
            static_cast<uint32_t>(float_to_q16(cos_angle)),  // rt_arg[3]: cos_angle (Q16.16)
            static_cast<uint32_t>(float_to_q16(sin_angle)),  // rt_arg[4]: sin_angle (Q16.16)
            static_cast<uint32_t>(float_to_q16(center_x)),   // rt_arg[5]: center_x (Q16.16)
            static_cast<uint32_t>(float_to_q16(center_y)),   // rt_arg[6]: center_y (Q16.16)
            static_cast<uint32_t>(fill_value_bf16)           // rt_arg[7]: fill_value (bfloat16)
        };

        // Writer runtime args
        std::vector<uint32_t> writer_runtime_args = {
            output_tensor.buffer()->address(),  // rt_arg[0]: output_buffer_address
            num_sticks,                         // rt_arg[1]: num_sticks
            sticks_processed,                   // rt_arg[2]: start_stick_id
        };

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        sticks_processed += num_sticks;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y,
         .enable_split_reader = false}};
}

void RotateDeviceOperation::NearestProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = output.buffer();

    // Recalculate rotation parameters
    const float angle_rad = operation_attributes.angle * M_PI / 180.0f;
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);

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

        // Update reader kernel runtime arguments
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();                           // input_buffer_address
            runtime_args[3] = static_cast<uint32_t>(float_to_q16(cos_angle));  // cos_angle (Q16.16)
            runtime_args[4] = static_cast<uint32_t>(float_to_q16(sin_angle));  // sin_angle (Q16.16)
            runtime_args[5] = static_cast<uint32_t>(float_to_q16(center_x));   // center_x (Q16.16)
            runtime_args[6] = static_cast<uint32_t>(float_to_q16(center_y));   // center_y (Q16.16)
            runtime_args[7] = static_cast<uint32_t>(fill_value_bf16);          // fill_value
        }

        // Update writer kernel runtime arguments
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();  // output_buffer_address
        }
    }
}

}  // namespace ttnn::operations::rotate
