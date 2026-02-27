// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/pool/rotate/device/rotate_device_operation.hpp>
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>
#include <ttnn/operations/pool/pool_utils.hpp>

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
    const bool is_sharded = input_tensor.is_sharded();

    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::tt_metal::IDevice* const device = output_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_batch = input_shape[0];
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];
    const uint32_t input_channels = input_shape[3];

    const float angle_rad = operation_attributes.angle * M_PI / 180.0f;
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);

    float center_x, center_y;
    if (operation_attributes.center.has_value()) {
        center_x = std::get<0>(operation_attributes.center.value()) - 0.5f;
        center_y = std::get<1>(operation_attributes.center.value()) - 0.5f;
    } else {
        center_x = (static_cast<float>(input_width) - 1.0f) / 2.0f;
        center_y = (static_cast<float>(input_height) - 1.0f) / 2.0f;
    }

    const uint16_t fill_value_bf16 = nearest_float_to_bfloat16(operation_attributes.fill);
    const uint32_t total_output_sticks = input_batch * input_height * input_width;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t input_stick_nbytes = input_channels * element_size;

    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1, core_group_2;
    uint32_t num_cores = 0;
    uint32_t num_sticks_per_core_group_1 = 0, num_sticks_per_core_group_2 = 0;
    std::vector<CoreCoord> logical_cores;
    uint32_t input_nsticks_per_core = 0;
    uint32_t output_nsticks_per_core = 0;
    bool is_block_sharded = false;
    bool is_width_sharded = false;
    uint32_t num_cores_x = 0;
    uint32_t shard_width = 0;

    const bool is_nd_sharded = input_tensor.memory_config().nd_shard_spec().has_value();

    if (is_sharded && !is_nd_sharded) {
        const auto input_shard_spec = input_tensor.shard_spec().value();
        all_cores = input_shard_spec.grid;
        num_cores = input_shard_spec.num_cores();
        input_nsticks_per_core = input_shard_spec.shape[0];
        output_nsticks_per_core = output_tensor.shard_spec().value().shape[0];
        logical_cores = corerange_to_cores(
            all_cores, num_cores, input_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        is_block_sharded =
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
        is_width_sharded =
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
        TT_FATAL(!is_width_sharded, "Width sharding is not supported for rotate operation");
        num_cores_x = input_shard_spec.grid.bounding_box().grid_size().x;
        shard_width = input_shard_spec.shape[1];
    } else if (is_nd_sharded) {
        const auto& nd_shard_spec = input_tensor.memory_config().nd_shard_spec().value();
        all_cores = nd_shard_spec.grid;
        num_cores = nd_shard_spec.grid.num_cores();
        const auto& shard_shape = nd_shard_spec.shard_shape;
        input_nsticks_per_core = shard_shape[-3] * shard_shape[-2];
        output_nsticks_per_core = input_nsticks_per_core;
        logical_cores = corerange_to_cores(
            all_cores, num_cores, nd_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        shard_width = shard_shape[-1];
    } else {
        const auto compute_grid_size = device->compute_with_storage_grid_size();
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);
        std::tie(num_cores, all_cores, core_group_1, core_group_2) =
            std::make_tuple(num_cores_used, all_cores_range, core_group_1_range, core_group_2_range);
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
        logical_cores = corerange_to_cores(all_cores, num_cores, true);
    }

    const uint32_t num_cores_y = device->compute_with_storage_grid_size().y;

    const bool any_sharded = is_sharded || is_nd_sharded;
    const uint32_t effective_channels = any_sharded ? shard_width : input_channels;
    const uint32_t aligned_input_stick_nbytes = any_sharded ? effective_channels * input_tensor.element_size()
                                                            : pool::get_aligned_stick_size(input_shape, input_tensor);
    const uint32_t aligned_output_stick_nbytes = any_sharded ? effective_channels * output_tensor.element_size()
                                                             : pool::get_aligned_stick_size(input_shape, output_tensor);

    const uint32_t available_l1 = NUM_TILES_DEST * tt::constants::TILE_HW * element_size;
    const uint32_t l1_for_cb = available_l1 / NEAREST_BUFFERING_FACTOR;
    const uint32_t max_cb_pages_from_l1 = l1_for_cb / aligned_input_stick_nbytes;

    const uint32_t max_sticks_per_core =
        any_sharded ? input_nsticks_per_core : std::max(num_sticks_per_core_group_1, num_sticks_per_core_group_2);
    uint32_t num_cb_pages = std::min(max_sticks_per_core, max_cb_pages_from_l1);

    uint32_t next_cb_index = tt::CBIndex::c_0;
    const uint32_t output_cb_page_size = aligned_input_stick_nbytes;

    auto [fill_cb_index, fill_cb_handle] =
        tt::tt_metal::create_cb(next_cb_index++, program, all_cores, output_cb_page_size, 1, output_cb_data_format);

    tt::tt_metal::CBHandle input_cb_handle = 0;
    uint32_t input_cb_index = 0;
    if (any_sharded) {
        std::tie(input_cb_index, input_cb_handle) = tt::tt_metal::create_cb(
            next_cb_index++,
            program,
            all_cores,
            aligned_input_stick_nbytes,
            input_nsticks_per_core,
            input_cb_data_format,
            input_tensor.buffer());
    }

    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++,
        program,
        all_cores,
        output_cb_page_size,
        any_sharded ? output_nsticks_per_core : num_cb_pages * NEAREST_BUFFERING_FACTOR,
        output_cb_data_format,
        any_sharded ? output_tensor.buffer() : nullptr);

    const bool fill_is_zero = (fill_value_bf16 == 0);
    const uint32_t batch_size = num_cb_pages < MAX_BATCH_SIZE ? num_cb_pages : MAX_BATCH_SIZE;

    const uint32_t effective_stick_nbytes = any_sharded ? effective_channels * element_size : input_stick_nbytes;

    std::vector<uint32_t> reader_compile_time_args = {
        output_cb_index,
        aligned_input_stick_nbytes,
        input_batch,
        input_height,
        input_width,
        effective_channels,
        num_cb_pages,
        fill_cb_index,
        effective_stick_nbytes,
        static_cast<uint32_t>(fill_is_zero),
        batch_size,
    };

    auto* input_buffer = input_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "Input tensor must be allocated on device for rotate operation");
    tt::tt_metal::TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_index,
        aligned_output_stick_nbytes,
        num_cb_pages,
        batch_size,
    };

    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(output_buffer != nullptr, "Output tensor must be allocated on device for rotate operation");
    tt::tt_metal::TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/rotate/device/kernels/dataflow/"
        "reader_rotate_nearest_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/rotate/device/kernels/dataflow/"
        "writer_rotate_nearest_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    if (any_sharded) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];

            uint32_t start_stick_id;
            if (is_width_sharded) {
                start_stick_id = 0;
            } else if (is_block_sharded) {
                uint32_t core_y = i / num_cores_x;
                start_stick_id = core_y * input_nsticks_per_core;
            } else {
                start_stick_id = i * input_nsticks_per_core;
            }

            std::vector<uint32_t> reader_runtime_args = {
                input_tensor.buffer()->address(),
                input_nsticks_per_core,
                start_stick_id,
                static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(cos_angle)),
                static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(sin_angle)),
                static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_x)),
                static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_y)),
                static_cast<uint32_t>(fill_value_bf16),
            };

            std::vector<uint32_t> writer_runtime_args = {
                output_tensor.buffer()->address(),
                input_nsticks_per_core,
                start_stick_id,
            };

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }
    } else {
        uint32_t sticks_processed = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t num_sticks =
                core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

            std::vector<uint32_t> reader_runtime_args = {
                input_tensor.buffer()->address(),
                num_sticks,
                sticks_processed,
                static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(cos_angle)),
                static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(sin_angle)),
                static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_x)),
                static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_y)),
                static_cast<uint32_t>(fill_value_bf16),
            };

            std::vector<uint32_t> writer_runtime_args = {
                output_tensor.buffer()->address(),
                num_sticks,
                sticks_processed,
            };

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

            sticks_processed += num_sticks;
        }
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y,
         .is_sharded = is_sharded,
         .logical_cores = logical_cores,
         .input_cb_handle = input_cb_handle,
         .output_cb_handle = output_cb_handle}};
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
    auto& is_sharded = cached_program.shared_variables.is_sharded;
    auto& logical_cores = cached_program.shared_variables.logical_cores;
    auto& input_cb_handle = cached_program.shared_variables.input_cb_handle;
    auto& output_cb_handle = cached_program.shared_variables.output_cb_handle;

    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = output.buffer();

    TT_FATAL(src_buffer != nullptr, "Input tensor buffer must not be null in override_runtime_arguments");
    TT_FATAL(dst_buffer != nullptr, "Output tensor buffer must not be null in override_runtime_arguments");

    const float angle_rad = operation_attributes.angle * M_PI / 180.0f;
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);

    const auto& input_shape = tensor_args.input.padded_shape();
    const uint32_t input_width = input_shape[2];
    const uint32_t input_height = input_shape[1];

    float center_x, center_y;
    if (operation_attributes.center.has_value()) {
        center_x = std::get<0>(operation_attributes.center.value()) - 0.5f;
        center_y = std::get<1>(operation_attributes.center.value()) - 0.5f;
    } else {
        center_x = (static_cast<float>(input_width) - 1.0f) / 2.0f;
        center_y = (static_cast<float>(input_height) - 1.0f) / 2.0f;
    }

    const uint16_t fill_value_bf16 = nearest_float_to_bfloat16(operation_attributes.fill);

    if (is_sharded) {
        tt::tt_metal::UpdateDynamicCircularBufferAddress(program, input_cb_handle, *src_buffer);
        tt::tt_metal::UpdateDynamicCircularBufferAddress(program, output_cb_handle, *dst_buffer);

        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];

            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[3] = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(cos_angle));
            runtime_args[4] = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(sin_angle));
            runtime_args[5] = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_x));
            runtime_args[6] = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_y));
            runtime_args[7] = static_cast<uint32_t>(fill_value_bf16);

            auto& writer_args = GetRuntimeArgs(program, writer_kernel_id, core);
            writer_args[0] = dst_buffer->address();
        }
    } else {
        for (uint32_t i = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[3] = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(cos_angle));
            runtime_args[4] = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(sin_angle));
            runtime_args[5] = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_x));
            runtime_args[6] = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_y));
            runtime_args[7] = static_cast<uint32_t>(fill_value_bf16);

            auto& writer_args = GetRuntimeArgs(program, writer_kernel_id, core);
            writer_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::rotate
