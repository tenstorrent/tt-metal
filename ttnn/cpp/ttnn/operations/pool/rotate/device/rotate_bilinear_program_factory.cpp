// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/pool/rotate/device/rotate_device_operation.hpp>
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>
#include <ttnn/operations/pool/pool_utils.hpp>
#include <ttnn/operations/pool/grid_sample/device/grid_sample_utils.hpp>

#include <cmath>
#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <ttnn/operations/cb_utils.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::rotate {

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
constexpr uint32_t BUFFERING_FACTOR = 2;
constexpr uint32_t REDUCTION_SIZE = 4;
constexpr uint32_t MAX_ROWS_FOR_REDUCTION = 16;
constexpr bool ONE_SCALAR_PER_CORE = false;
constexpr uint32_t DUMMY_CB_ID = 32;

static uint16_t float_to_bfloat16(float value) {
    bfloat16 bf16_value(value);
    return std::bit_cast<uint16_t>(bf16_value);
}

RotateDeviceOperation::BilinearProgramFactory::cached_program_t RotateDeviceOperation::BilinearProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = output;

    tt::tt_metal::Program program{};

    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::tt_metal::IDevice* const device = output_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();
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

    const bool is_bfloat16 = input_tensor.dtype() == DataType::BFLOAT16;
    uint32_t fill_value_bits;
    if (is_bfloat16) {
        fill_value_bits = float_to_bfloat16(operation_attributes.fill);
    } else {
        fill_value_bits = std::bit_cast<uint32_t>(operation_attributes.fill);
    }

    const uint32_t total_output_sticks = input_batch * input_height * input_width;
    const bool is_input_sharded = input_tensor.is_sharded();
    const bool is_output_sharded = output_tensor.is_sharded();

    const auto compute_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_grid_size.y;

    tt::tt_metal::CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores = 0;
    uint32_t num_sticks_per_core_group_1 = 0, num_sticks_per_core_group_2 = 0;
    uint32_t input_nsticks_per_core = 0;
    uint32_t output_nsticks_per_core = 0;
    std::vector<CoreCoord> logical_cores;
    bool is_block_sharded = false;
    bool is_width_sharded = false;
    uint32_t num_cores_x = 0;

    if (is_input_sharded) {
        const auto input_shard_spec = input_tensor.shard_spec().value();
        all_cores = input_shard_spec.grid;
        num_cores = input_shard_spec.num_cores();
        input_nsticks_per_core = input_shard_spec.shape[0];
        output_nsticks_per_core =
            is_output_sharded ? output_tensor.shard_spec().value().shape[0] : input_nsticks_per_core;
        logical_cores = corerange_to_cores(
            all_cores, num_cores, input_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        is_block_sharded =
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
        is_width_sharded =
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
        num_cores_x = input_shard_spec.grid.bounding_box().grid_size().x;
        num_sticks_per_core_group_1 = input_nsticks_per_core;
        core_group_1 = all_cores;
    } else if (is_output_sharded) {
        const auto output_shard_spec = output_tensor.shard_spec().value();
        output_nsticks_per_core = output_shard_spec.shape[0];
        num_cores = (total_output_sticks + output_nsticks_per_core - 1) / output_nsticks_per_core;
        all_cores = output_shard_spec.grid;
        logical_cores = corerange_to_cores(
            all_cores, num_cores, output_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        num_sticks_per_core_group_1 = output_nsticks_per_core;
        core_group_1 = all_cores;
    } else {
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);

        num_cores = num_cores_used;
        all_cores = all_cores_range;
        core_group_1 = core_group_1_range;
        core_group_2 = core_group_2_range;
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
        logical_cores = corerange_to_cores(all_cores, num_cores, true);
    }

    const bool any_sharded = is_input_sharded || is_output_sharded;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t input_stick_nbytes = input_channels * element_size;

    uint32_t cb_idx = tt::CBIndex::c_0;

    const uint32_t fill_cb_page_size = input_stick_nbytes;
    const auto [fill_cb_index, fill_cb_handle] =
        tt::tt_metal::create_cb(cb_idx++, program, all_cores, fill_cb_page_size, 1, input_cb_data_format);

    const uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_channels / tt::constants::TILE_WIDTH);
    const uint32_t input_cb_page_size = in_ntiles_c * tt::constants::TILE_HW * element_size;
    const auto [input_cb_index, input_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++, program, all_cores, input_cb_page_size, BUFFERING_FACTOR, input_cb_data_format);

    cb_idx++;

    const uint32_t scalar_cb_page_size = tt::tile_size(input_cb_data_format);
    const auto [scalar_cb_index, scalar_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++, program, all_cores, scalar_cb_page_size, BUFFERING_FACTOR, input_cb_data_format);

    cb_idx++;

    const uint32_t out_ntiles_c = (uint32_t)std::ceil((float)output_shape[-1] / tt::constants::FACE_WIDTH);
    const uint32_t output_cb_page_size = tt::constants::FACE_WIDTH * element_size;
    const uint32_t output_cb_pages =
        any_sharded ? output_nsticks_per_core * out_ntiles_c : out_ntiles_c * BUFFERING_FACTOR;
    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++,
        program,
        all_cores,
        output_cb_page_size,
        output_cb_pages,
        output_cb_data_format,
        any_sharded ? output_tensor.buffer() : nullptr);

    const bool fill_is_zero = (fill_value_bits == 0);

    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index,
        scalar_cb_index,
        input_stick_nbytes,
        input_batch,
        input_height,
        input_width,
        fill_cb_index,
        input_channels,
        static_cast<uint32_t>(fill_is_zero),
        element_size,
    };

    auto* input_buffer = input_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "Input tensor must be allocated on device for rotate operation");
    tt::tt_metal::TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/rotate/device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    const uint32_t in_nblocks_c = (uint32_t)std::ceil((float)in_ntiles_c / MAX_TILES_PER_REDUCTION);

    auto create_compute_kernel = [&](tt::tt_metal::CoreRangeSet cores, uint32_t total_interpolations) {
        std::vector<uint32_t> compute_compile_time_args = {
            in_ntiles_c,
            REDUCTION_SIZE,
            0,
            total_interpolations,
            input_channels,
            in_nblocks_c,
            MAX_ROWS_FOR_REDUCTION,
            input_cb_index,
            DUMMY_CB_ID,
            scalar_cb_index,
            DUMMY_CB_ID,
            DUMMY_CB_ID,
            DUMMY_CB_ID,
            DUMMY_CB_ID,
            DUMMY_CB_ID,
            DUMMY_CB_ID,
            DUMMY_CB_ID,
            output_cb_index,
            DUMMY_CB_ID,
            ONE_SCALAR_PER_CORE ? 1U : 0U,
            DUMMY_CB_ID,
            0U,
            0U,
            0U,
            1U,
            1U,
            1U,
            1U,
            1U,
            1U,
            0U,
        };

        return tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp",
            cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = compute_compile_time_args,
                .defines = pool::get_defines(pool::Pool2DType::AVG_POOL2D)});
    };

    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    if (any_sharded || core_group_1.num_cores() > 0) {
        compute_kernel_id = create_compute_kernel(
            any_sharded ? all_cores : core_group_1,
            any_sharded ? output_nsticks_per_core : num_sticks_per_core_group_1);
    }
    if (!any_sharded && core_group_2.num_cores() > 0) {
        create_compute_kernel(core_group_2, num_sticks_per_core_group_2);
    }

    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    if (!any_sharded) {
        const uint32_t output_stick_size = input_channels * element_size;
        std::vector<uint32_t> writer_compile_time_args = {output_cb_index, output_stick_size, out_ntiles_c};
        auto* output_buffer = output_tensor.buffer();
        TT_FATAL(output_buffer != nullptr, "Output tensor must be allocated on device for rotate operation");
        tt::tt_metal::TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);

        writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }

    const int32_t cos_angle_q16 = fixed_point_arithmetic::float_to_fixed(cos_angle);
    const int32_t sin_angle_q16 = fixed_point_arithmetic::float_to_fixed(sin_angle);
    const int32_t center_x_q16 = fixed_point_arithmetic::float_to_fixed(center_x);
    const int32_t center_y_q16 = fixed_point_arithmetic::float_to_fixed(center_y);

    uint32_t sticks_processed = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];
        uint32_t num_sticks;
        uint32_t start_stick_id;

        if (is_input_sharded) {
            num_sticks = input_nsticks_per_core;
            if (is_width_sharded) {
                start_stick_id = 0;
            } else if (is_block_sharded) {
                uint32_t core_y = i / num_cores_x;
                start_stick_id = core_y * input_nsticks_per_core;
            } else {
                start_stick_id = i * input_nsticks_per_core;
            }
        } else if (is_output_sharded) {
            num_sticks = output_nsticks_per_core;
            start_stick_id = sticks_processed;
        } else {
            num_sticks = core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;
            start_stick_id = sticks_processed;
        }

        std::vector<uint32_t> reader_runtime_args = {
            input_tensor.buffer()->address(),
            num_sticks,
            start_stick_id,
            static_cast<uint32_t>(cos_angle_q16),
            static_cast<uint32_t>(sin_angle_q16),
            static_cast<uint32_t>(center_x_q16),
            static_cast<uint32_t>(center_y_q16),
            static_cast<uint32_t>(fill_value_bits)};

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        if (!any_sharded) {
            std::vector<uint32_t> writer_runtime_args = {output_tensor.buffer()->address(), num_sticks, start_stick_id};
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }

        sticks_processed += num_sticks;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y,
         .is_input_sharded = is_input_sharded,
         .is_output_sharded = is_output_sharded,
         .logical_cores = logical_cores,
         .output_cb_handle = output_cb_handle}};
}

void RotateDeviceOperation::BilinearProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& is_input_sharded = cached_program.shared_variables.is_input_sharded;
    auto& is_output_sharded = cached_program.shared_variables.is_output_sharded;
    auto& logical_cores = cached_program.shared_variables.logical_cores;
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

    const bool is_bfloat16 = tensor_args.input.dtype() == DataType::BFLOAT16;
    uint32_t fill_value_bits;
    if (is_bfloat16) {
        fill_value_bits = float_to_bfloat16(operation_attributes.fill);
    } else {
        fill_value_bits = std::bit_cast<uint32_t>(operation_attributes.fill);
    }

    const int32_t cos_angle_q16 = fixed_point_arithmetic::float_to_fixed(cos_angle);
    const int32_t sin_angle_q16 = fixed_point_arithmetic::float_to_fixed(sin_angle);
    const int32_t center_x_q16 = fixed_point_arithmetic::float_to_fixed(center_x);
    const int32_t center_y_q16 = fixed_point_arithmetic::float_to_fixed(center_y);

    const bool any_sharded = is_input_sharded || is_output_sharded;
    if (any_sharded) {
        tt::tt_metal::UpdateDynamicCircularBufferAddress(program, output_cb_handle, *dst_buffer);
    }

    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[3] = static_cast<uint32_t>(cos_angle_q16);
            runtime_args[4] = static_cast<uint32_t>(sin_angle_q16);
            runtime_args[5] = static_cast<uint32_t>(center_x_q16);
            runtime_args[6] = static_cast<uint32_t>(center_y_q16);
            runtime_args[7] = fill_value_bits;
        }

        if (!any_sharded) {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::rotate
