// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

    const uint16_t fill_value_bf16 = float_to_bfloat16(operation_attributes.fill);

    const uint32_t total_output_sticks = input_batch * input_height * input_width;

    const auto compute_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);

    const uint32_t num_cores_y = compute_grid_size.y;

    std::vector<CoreCoord> logical_cores = corerange_to_cores(all_cores, num_cores, true);

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t input_stick_nbytes = input_channels * element_size;

    uint32_t cb_idx = tt::CBIndex::c_0;

    cb_idx++;

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
    const uint32_t output_cb_pages = out_ntiles_c * BUFFERING_FACTOR;
    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++, program, all_cores, output_cb_page_size, output_cb_pages, output_cb_data_format);

    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index,
        scalar_cb_index,
        input_stick_nbytes,
        input_batch,
        input_height,
        input_width,
    };

    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

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
    if (core_group_1.num_cores() > 0) {
        compute_kernel_id = create_compute_kernel(core_group_1, num_sticks_per_core_group_1);
    }
    if (core_group_2.num_cores() > 0) {
        create_compute_kernel(core_group_2, num_sticks_per_core_group_2);
    }

    const uint32_t output_stick_size = input_channels * element_size;
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index, output_stick_size, out_ntiles_c};
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t sticks_processed = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];
        const uint32_t num_sticks =
            core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

        const int32_t cos_angle_q16 = fixed_point_arithmetic::float_to_fixed(cos_angle);
        const int32_t sin_angle_q16 = fixed_point_arithmetic::float_to_fixed(sin_angle);
        const int32_t center_x_q16 = fixed_point_arithmetic::float_to_fixed(center_x);
        const int32_t center_y_q16 = fixed_point_arithmetic::float_to_fixed(center_y);

        std::vector<uint32_t> reader_runtime_args = {
            input_tensor.buffer()->address(),
            num_sticks,
            sticks_processed,
            static_cast<uint32_t>(cos_angle_q16),
            static_cast<uint32_t>(sin_angle_q16),
            static_cast<uint32_t>(center_x_q16),
            static_cast<uint32_t>(center_y_q16),
            static_cast<uint32_t>(fill_value_bf16)};

        std::vector<uint32_t> writer_runtime_args = {output_tensor.buffer()->address(), num_sticks, sticks_processed};

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        sticks_processed += num_sticks;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
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
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = output.buffer();

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

    const uint16_t fill_value_bf16 = float_to_bfloat16(operation_attributes.fill);

    const int32_t cos_angle_q16 = fixed_point_arithmetic::float_to_fixed(cos_angle);
    const int32_t sin_angle_q16 = fixed_point_arithmetic::float_to_fixed(sin_angle);
    const int32_t center_x_q16 = fixed_point_arithmetic::float_to_fixed(center_x);
    const int32_t center_y_q16 = fixed_point_arithmetic::float_to_fixed(center_y);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[3] = static_cast<uint32_t>(cos_angle_q16);
            runtime_args[4] = static_cast<uint32_t>(sin_angle_q16);
            runtime_args[5] = static_cast<uint32_t>(center_x_q16);
            runtime_args[6] = static_cast<uint32_t>(center_y_q16);
            runtime_args[7] = static_cast<uint32_t>(fill_value_bf16);
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::rotate
