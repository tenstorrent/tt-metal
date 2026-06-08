// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_last_dim.hpp"

#include <cstdint>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_common.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor RepeatProgramFactoryLastDim::create_descriptor(
    const RepeatParams& operation_attributes, const RepeatInputs& tensor_args, Tensor& tensor_return_value) {
    // We are repeating the last dim on a 2D shape
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const uint32_t num_repeats = operation_attributes.m_num_repeats;
    // get datum size
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t data_size = input.element_size();
    IDevice* device = input.device();
    // Multi device pre-computation
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;
    const CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    const CoreRangeSet total_core_ranges{total_cores};

    ttnn::Shape input_log_shape = ttnn::Shape(input.logical_shape().view());
    ttnn::Shape output_log_shape = ttnn::Shape(output.logical_shape().view());
    const uint32_t source_page_size_bytes = input_log_shape[-1] * data_size;
    const uint32_t dest_page_size_bytes = source_page_size_bytes * num_repeats;
    TT_FATAL(
        dest_page_size_bytes == output_log_shape[-1] * data_size,
        "Data size of output does not match requirement for repeat last dim");
    uint32_t read_start_page = 0;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Find how many input pages each core is responsible for so that we always start at the beginning of a read and
    // write page Since the logical volumes match, we are guaranteed that the very last page is aligned
    const uint32_t number_of_pages = input_log_shape[-2];
    const uint32_t responsibility = ((number_of_pages - 1) / num_cores_total) + 1;
    const uint32_t cb_size_bytes = READ_ALIGNMENT * 2 + (source_page_size_bytes & 0xF) == 0 ? source_page_size_bytes
                                   : (source_page_size_bytes & 0x7) == 0                    ? source_page_size_bytes * 2
                                   : (source_page_size_bytes & 0x3) == 0                    ? source_page_size_bytes * 4
                                   : (source_page_size_bytes & 0x1) == 0                    ? source_page_size_bytes * 8
                                                                         : source_page_size_bytes * 16;
    constexpr uint32_t src0_cb_index = 0;
    constexpr uint32_t src1_cb_index = 1;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_size_bytes,
        .core_ranges = total_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = cb_size_bytes,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_size_bytes,
        .core_ranges = total_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = cb_data_format,
            .page_size = cb_size_bytes,
        }}},
    });

    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t)source_page_size_bytes, (std::uint32_t)num_repeats, src0_cb_index, src1_cb_index};
    TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/repeat/device/kernels/repeat_last_dim_rm.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = total_core_ranges;
    reader_desc.compile_time_args = std::move(compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    uint32_t done = 0;
    for (uint32_t core_x = 0; core_x < num_cores_x; core_x++) {
        for (uint32_t core_y = 0; core_y < num_cores_y; core_y++) {
            const CoreCoord core = {core_x, core_y};
            if (done == 1) {
                // Buffer* args trigger BufferBinding entries; the framework patches their
                // addresses on cache hits without rebuilding the descriptor.
                reader_desc.emplace_runtime_args(core, {src_buffer, dst_buffer, uint32_t{0}, uint32_t{0}, uint32_t{1}});
            } else {
                const uint32_t start_of_read = read_start_page;
                uint32_t end_of_read = read_start_page + responsibility;
                end_of_read = end_of_read < number_of_pages ? end_of_read : number_of_pages;

                reader_desc.emplace_runtime_args(
                    core, {src_buffer, dst_buffer, start_of_read, end_of_read, uint32_t{0}});
                read_start_page = end_of_read;
                done = (end_of_read == input_log_shape[-2]) ? 1 : 0;
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::prim
