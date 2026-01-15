// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample_nearest_float_program_factory.hpp"

#include <ttnn/operations/pool/pool_utils.hpp>

#include <cmath>
#include <cstdint>
#include <tt-metalium/work_split.hpp>
#include <ttnn/operations/cb_utils.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "upsample/device/upsample_device_operation_types.hpp"

namespace ttnn::operations::pool::upsample::program {

constexpr uint32_t BUFFERING_FACTOR = 2;

UpsampleNearestFloatProgramFactory::cached_program_t UpsampleNearestFloatProgramFactory::create(
    const UpsampleParams& operation_attributes, const UpsampleInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;

    tt::tt_metal::Program program = tt::tt_metal::Program{};

    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    auto* const device = output_tensor.device();

    const auto& input_shape = input_tensor.logical_shape();
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];

    // Output dimensions (from logical shape)
    const uint32_t output_height =
        static_cast<uint32_t>(std::floor(input_height * operation_attributes.scale_factor_h));
    const uint32_t output_width = static_cast<uint32_t>(std::floor(input_width * operation_attributes.scale_factor_w));

    // Calculate reciprocal scale factors for kernel (fixed-point Q16.16)
    // src = floor(dst / scale) = floor(dst * reciprocal_scale)
    // We need to round UP the reciprocal to ensure boundary values are handled correctly.
    // For example, with scale=3, dst=3: we need floor(3/3)=1, which requires 3*(1/3) >= 1.0
    // Rounding up the reciprocal ensures this property is maintained.
    constexpr int32_t FIXED_ONE = 1 << 16;
    const float reciprocal_scale_h = 1.0f / operation_attributes.scale_factor_h;
    const float reciprocal_scale_w = 1.0f / operation_attributes.scale_factor_w;
    const int32_t reciprocal_scale_h_fixed = static_cast<int32_t>(std::ceil(reciprocal_scale_h * FIXED_ONE));
    const int32_t reciprocal_scale_w_fixed = static_cast<int32_t>(std::ceil(reciprocal_scale_w * FIXED_ONE));

    // Work distribution - Total work units = N * H_out * W_out (one output stick per work unit)

    const uint32_t total_pages_in_output = output_tensor.buffer()->num_pages();

    const tt::tt_metal::Shape& output_shape = output_tensor.padded_shape();

    const uint32_t num_pages_across_width =
        total_pages_in_output / (output_shape[0] * output_shape[1] * output_shape[2]);

    const uint32_t aligned_input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t aligned_output_page_size = output_tensor.buffer()->aligned_page_size();

    const uint32_t input_page_size = input_tensor.buffer()->page_size();
    const uint32_t output_page_size = output_tensor.buffer()->page_size();

    TT_FATAL(
        input_page_size == output_page_size,
        "Input and output page sizes must match for nearest upsample, got input_page_size={} output_page_size={}",
        input_page_size,
        output_page_size);

    const tt::tt_metal::CoreCoord compute_grid_size = device->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, total_pages_in_output);

    const std::vector<tt::tt_metal::CoreCoord> logical_cores =
        tt::tt_metal::corerange_to_cores(all_cores, std::nullopt, true);

    // Calculate stick sizes (aligned based on buffer type for efficient reads)

    const uint32_t num_cb_pages = BUFFERING_FACTOR;

    uint32_t next_cb_index = tt::CBIndex::c_0;
    const uint32_t output_cb_page_size = aligned_output_page_size;

    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        next_cb_index++,
        program,
        all_cores,
        output_cb_page_size,
        num_cb_pages * BUFFERING_FACTOR,
        output_cb_data_format);

    std::vector<uint32_t> reader_compile_time_args = {
        output_cb_index,
        aligned_input_page_size,
        input_height,
        input_width,
        output_height,
        output_width,
        num_pages_across_width,
        static_cast<uint32_t>(reciprocal_scale_h_fixed),
        static_cast<uint32_t>(reciprocal_scale_w_fixed),
    };

    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_index,
        aligned_output_page_size,
    };

    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

    const tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/"
        "reader_upsample_nearest_float.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    const tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/"
        "writer_upsample_nearest_float.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Set runtime arguments for each core
    uint32_t sticks_processed = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const tt::tt_metal::CoreCoord& core = logical_cores[i];
        const uint32_t num_sticks =
            core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

        // Reader runtime args
        std::vector<uint32_t> reader_runtime_args = {
            input_tensor.buffer()->address(),  // rt_arg[0]: input_buffer_address
            num_sticks,                        // rt_arg[1]: num_sticks
            sticks_processed,                  // rt_arg[2]: start_stick_id
        };

        // Writer runtime args
        std::vector<uint32_t> writer_runtime_args = {
            output_tensor.buffer()->address(),  // rt_arg[0]: output_buffer_address
            num_sticks,                         // rt_arg[1]: num_sticks
            sticks_processed,                   // rt_arg[2]: start_stick_id
        };

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, std::move(reader_runtime_args));
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, std::move(writer_runtime_args));

        sticks_processed += num_sticks;
    }

    return {
        std::move(program),
        UpsampleNearestFloatSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = 0,  // No compute kernel for this data movement op
            .writer_kernel_id = writer_kernel_id,
            .all_cores = all_cores,
            .num_cores = num_cores}};
}

void UpsampleNearestFloatProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const UpsampleParams& operation_attributes,
    const UpsampleInputs& tensor_args,
    Tensor& output_tensor) {
    tt::tt_metal::Program& program = cached_program.program;
    const tt::tt_metal::KernelHandle reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const tt::tt_metal::KernelHandle writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const uint32_t num_cores = cached_program.shared_variables.num_cores;

    const tt::tt_metal::Buffer* const src_buffer = tensor_args.input_tensor.buffer();
    const tt::tt_metal::Buffer* const dst_buffer = output_tensor.buffer();

    (void)operation_attributes;

    auto* const device = tensor_args.input_tensor.device();
    const tt::tt_metal::CoreCoord compute_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_grid_size.y;

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
}

}  // namespace ttnn::operations::pool::upsample::program
