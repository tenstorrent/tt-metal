// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample_device_operation.hpp"

#include <cmath>
#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <ttnn/operations/pool/pool_utils.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

constexpr uint32_t BUFFERING_FACTOR = 2;

ProgramDescriptor UpsampleNearestFloatProgramFactory::create_descriptor(
    const UpsampleParams& operation_attributes, const Tensor& input, Tensor& output_tensor) {
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    auto* const device = output_tensor.device();

    const auto& input_shape = input.logical_shape();
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

    const Shape& output_shape = output_tensor.padded_shape();

    const uint32_t num_pages_across_width =
        total_pages_in_output / (output_shape[0] * output_shape[1] * output_shape[2]);

    const uint32_t aligned_input_page_size = input.buffer()->aligned_page_size();
    const uint32_t aligned_output_page_size = output_tensor.buffer()->aligned_page_size();

    const uint32_t input_page_size = input.buffer()->page_size();
    const uint32_t output_page_size = output_tensor.buffer()->page_size();

    TT_FATAL(
        input_page_size == output_page_size,
        "Input and output page sizes must match for nearest upsample, got input_page_size={} output_page_size={}",
        input_page_size,
        output_page_size);

    const CoreCoord compute_grid_size = device->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
            split_work_to_cores(compute_grid_size, total_pages_in_output);

    const std::vector<CoreCoord> logical_cores = corerange_to_cores(all_cores, std::nullopt, true);

    // Calculate stick sizes (aligned based on buffer type for efficient reads)
    const uint32_t num_cb_pages = BUFFERING_FACTOR;

    uint32_t next_cb_index = tt::CBIndex::c_0;
    const uint32_t output_cb_page_size = aligned_output_page_size;

    ProgramDescriptor desc;

    const uint32_t output_cb_index = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_page_size * num_cb_pages * BUFFERING_FACTOR,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_cb_page_size,
        }}},
    });

    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {
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
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {
        output_cb_index,
        aligned_output_page_size,
    };
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_upsample_nearest_float.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_nearest_float.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Set runtime arguments for each core
    uint32_t sticks_processed = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];
        const uint32_t num_sticks =
            core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

        reader_desc.emplace_runtime_args(
            core,
            {
                input.buffer(),    // rt_arg[0]: input_buffer_address
                num_sticks,        // rt_arg[1]: num_sticks
                sticks_processed,  // rt_arg[2]: start_stick_id
            });
        writer_desc.emplace_runtime_args(
            core,
            {
                output_tensor.buffer(),  // rt_arg[0]: output_buffer_address
                num_sticks,              // rt_arg[1]: num_sticks
                sticks_processed,        // rt_arg[2]: start_stick_id
            });

        sticks_processed += num_sticks;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
