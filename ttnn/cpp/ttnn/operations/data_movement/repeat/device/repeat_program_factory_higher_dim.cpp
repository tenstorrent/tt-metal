// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_higher_dim.hpp"

#include <cstdint>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_common.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor RepeatProgramFactoryHigherDim::create_descriptor(
    const RepeatParams& operation_attributes, const RepeatInputs& tensor_args, Tensor& tensor_return_value) {
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

    uint32_t page_size_bytes;
    uint32_t number_of_higher_pages;
    uint32_t number_of_lower_pages;
    uint32_t number_of_rep_dim_pages;

    if (operation_attributes.m_tile_page_size_bytes > 0) {
        // Tile-native: host supplies tile-space page counts.
        page_size_bytes = operation_attributes.m_tile_page_size_bytes;
        number_of_higher_pages = operation_attributes.m_tile_higher_pages;
        number_of_rep_dim_pages = operation_attributes.m_tile_rep_dim_pages;
        number_of_lower_pages = operation_attributes.m_tile_lower_pages;
    } else {
        page_size_bytes = input_log_shape[3] * data_size;
        TT_FATAL(
            page_size_bytes == output_log_shape[3] * data_size,
            "Data size of output does not match requirement for repeat higher dim");
        // Per-core page count so read/write start on page boundaries.
        number_of_higher_pages = input_log_shape[0];
        number_of_rep_dim_pages = input_log_shape[1];
        number_of_lower_pages = input_log_shape[2];
    }
    uint32_t read_start_page = 0;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    const uint32_t cb_size_bytes = (READ_ALIGNMENT * 2) + page_size_bytes;
    constexpr uint32_t src0_cb_index = 0;
    constexpr uint32_t src1_cb_index = 1;

    // TILE/sharded -> tile; RM sharded -> rm_sharded; RM interleaved -> rm_interleaved.
    const bool is_tile_native = operation_attributes.m_tile_page_size_bytes > 0;
    const bool src_sharded = src_buffer->buffer_distribution_spec().has_value();
    const bool dst_sharded = dst_buffer->buffer_distribution_spec().has_value();
    const bool needs_alignment_cb = !is_tile_native && !src_sharded && !dst_sharded;

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

    // Second CB only for interleaved RM (alignment scratchpad).
    if (needs_alignment_cb) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb_size_bytes,
            .core_ranges = total_core_ranges,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = src1_cb_index,
                .data_format = cb_data_format,
                .page_size = cb_size_bytes,
            }}},
        });
    }

    const char* kernel_source = nullptr;
    std::vector<uint32_t> compile_time_args;
    if (is_tile_native) {
        kernel_source = "ttnn/cpp/ttnn/operations/data_movement/repeat/device/kernels/repeat_higher_dim_tile.cpp";
        compile_time_args = {
            (std::uint32_t)page_size_bytes, src0_cb_index, number_of_lower_pages, number_of_rep_dim_pages};
    } else if (src_sharded || dst_sharded) {
        kernel_source = "ttnn/cpp/ttnn/operations/data_movement/repeat/device/kernels/repeat_higher_dim_rm_sharded.cpp";
        compile_time_args = {
            (std::uint32_t)page_size_bytes, src0_cb_index, number_of_lower_pages, number_of_rep_dim_pages};
    } else {
        kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/repeat/device/kernels/repeat_higher_dim_rm_interleaved.cpp";
        compile_time_args = {
            (std::uint32_t)page_size_bytes,
            src0_cb_index,
            src1_cb_index,
            number_of_lower_pages,
            number_of_rep_dim_pages};
    }

    std::vector<uint32_t> common_runtime_args;
    // RuntimeTensorShape: safe for interleaved; sharded moves shape to runtime args.
    TensorAccessorArgs(*src_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(compile_time_args, common_runtime_args);
    TensorAccessorArgs(*dst_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(compile_time_args, common_runtime_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kernel_source;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = total_core_ranges;
    reader_desc.compile_time_args = std::move(compile_time_args);
    reader_desc.common_runtime_args = std::move(common_runtime_args);
    reader_desc.config = ReaderConfigDescriptor{};

    uint32_t done = 0;
    // Determine runtime arguments
    const bool divide_on_higher = number_of_higher_pages > number_of_lower_pages;

    const uint32_t responsibility_chunk =
        (divide_on_higher ? number_of_higher_pages : number_of_lower_pages) / num_cores_total;
    const uint32_t responsibility_mod =
        (divide_on_higher ? number_of_higher_pages : number_of_lower_pages) % num_cores_total;
    uint32_t core_count = 0;
    for (uint32_t core_x = 0; core_x < num_cores_x; core_x++) {
        for (uint32_t core_y = 0; core_y < num_cores_y; core_y++) {
            const uint32_t responsibility =
                core_count++ < responsibility_mod ? responsibility_chunk + 1 : responsibility_chunk;
            const CoreCoord core = {core_x, core_y};
            if (done == 1) {
                // Idle core: zero args + early exit.
                reader_desc.emplace_runtime_args(
                    core,
                    {uint32_t{0},
                     uint32_t{0},
                     uint32_t{0},
                     uint32_t{0},
                     uint32_t{0},
                     uint32_t{0},
                     uint32_t{0},
                     uint32_t{1}});
            } else if (divide_on_higher) {
                const uint32_t start_of_read = read_start_page;
                uint32_t end_of_read = read_start_page + responsibility;
                end_of_read = end_of_read < number_of_higher_pages ? end_of_read : number_of_higher_pages;

                // Buffer* runtime args patch on program cache hit.
                reader_desc.emplace_runtime_args(
                    core,
                    {src_buffer,
                     dst_buffer,
                     start_of_read,
                     end_of_read,
                     uint32_t{0},
                     number_of_lower_pages,
                     num_repeats,
                     uint32_t{0}});
                read_start_page = end_of_read;
                done = (end_of_read == number_of_higher_pages) ? 1 : 0;
            } else {
                const uint32_t start_of_read = read_start_page;
                uint32_t end_of_read = read_start_page + responsibility;
                end_of_read = end_of_read < number_of_lower_pages ? end_of_read : number_of_lower_pages;

                reader_desc.emplace_runtime_args(
                    core,
                    {src_buffer,
                     dst_buffer,
                     uint32_t{0},
                     number_of_higher_pages,
                     start_of_read,
                     end_of_read,
                     num_repeats,
                     uint32_t{0}});
                read_start_page = end_of_read;
                done = (end_of_read == number_of_lower_pages) ? 1 : 0;
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::prim
