// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_stride.hpp"

#include <optional>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SliceRmStrideProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    tt::tt_metal::IDevice* device = input_tensor.device();
    ProgramDescriptor desc;

    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output.padded_shape();
    uint32_t element_size = input_tensor.element_size();

    // Calculate total output rows based on tensor rank
    uint32_t total_output_rows = output_shape.volume() / output_shape[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), total_output_rows)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_output_rows);

    // Select kernels based on tensor rank
    std::string reader_kernel_path;
    std::string writer_kernel_path;
    if (input_shape.rank() <= 4) {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_4d.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_4d.cpp";
    } else {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_nd.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_nd.cpp";
    }

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t actual_input_w = input_shape[-1];
    uint32_t input_bytes_per_row = actual_input_w * element_size;
    uint32_t cb_page_size = input_bytes_per_row;

    auto src_buffer_alignment = input_tensor.buffer()->alignment();
    auto dst_buffer_alignment = output.buffer()->alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    uint32_t cb_page_size_aligned = tt::round_up(cb_page_size, alignment);
    uint32_t cb_total_size = 2 * cb_page_size_aligned;

    constexpr uint8_t in_cb = 0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_total_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in_cb,
            .data_format = cb_data_format,
            .page_size = cb_page_size_aligned,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args = {in_cb, element_size};
    TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {in_cb, element_size};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_path;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Calculate runtime arguments
    uint32_t tensor_rank = input_shape.rank();
    uint32_t base_rows_per_core = total_output_rows / num_cores;
    uint32_t extra_rows = total_output_rows % num_cores;

    const bool using_4d_kernels = input_shape.rank() <= 4;
    const auto& slice_start = args.slice_start;
    const auto& slice_end = args.slice_end;
    const auto& slice_step = args.step;

    auto all_cores_vec = corerange_to_cores(all_cores);
    reader_desc.runtime_args.reserve(all_cores_vec.size());
    writer_desc.runtime_args.reserve(all_cores_vec.size());

    uint32_t row_start_id = 0;
    uint32_t extra_rows_remaining = extra_rows;

    for (uint32_t core_idx = 0; core_idx < all_cores_vec.size(); ++core_idx) {
        uint32_t rows_for_this_core = base_rows_per_core;
        if (extra_rows_remaining > 0) {
            rows_for_this_core += 1;
            extra_rows_remaining -= 1;
        }

        std::vector<uint32_t> reader_args;
        std::vector<uint32_t> writer_args;

        if (using_4d_kernels) {
            reader_args = {
                input_tensor.buffer()->address(),
                tensor_rank,
                input_shape[-1],
                input_shape[-2],
                input_shape[-3],
                input_shape[-4],
                output_shape[-1],
                output_shape[-2],
                output_shape[-3],
                output_shape[-4],
                slice_start[-1],
                slice_end[-1],
                slice_step[-1],
                slice_start[-2],
                slice_end[-2],
                slice_step[-2],
                slice_start[-3],
                slice_end[-3],
                slice_step[-3],
                slice_start[-4],
                slice_end[-4],
                slice_step[-4],
                element_size,
                rows_for_this_core,
                row_start_id};

            writer_args = {
                output.buffer()->address(),
                tensor_rank,
                output_shape[-1],
                output_shape[-2],
                output_shape[-3],
                output_shape[-4],
                element_size,
                rows_for_this_core,
                row_start_id};
        } else {
            reader_args = {
                input_tensor.buffer()->address(), tensor_rank, element_size, rows_for_this_core, row_start_id};
            reader_args.insert(reader_args.end(), input_shape.cbegin(), input_shape.cend());
            reader_args.insert(reader_args.end(), output_shape.cbegin(), output_shape.cend());
            reader_args.insert(reader_args.end(), slice_start.cbegin(), slice_start.cend());
            reader_args.insert(reader_args.end(), slice_end.cbegin(), slice_end.cend());
            reader_args.insert(reader_args.end(), slice_step.cbegin(), slice_step.cend());

            writer_args = {output.buffer()->address(), tensor_rank, element_size, rows_for_this_core, row_start_id};
            writer_args.insert(writer_args.end(), output_shape.cbegin(), output_shape.cend());
        }

        reader_desc.runtime_args.emplace_back(all_cores_vec[core_idx], std::move(reader_args));
        writer_desc.runtime_args.emplace_back(all_cores_vec[core_idx], std::move(writer_args));
        row_start_id += rows_for_this_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
