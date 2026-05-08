// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_device_operation.hpp"

#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr const char* KERNEL_READER =
    "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/redistribute_pages_row_major_reader.cpp";
constexpr const char* KERNEL_WRITER =
    "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/redistribute_pages_row_major_writer.cpp";

}  // namespace

ProgramDescriptor CopyDeviceOperation::DefaultRowMajor::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input;
    const auto& output = output_tensor;

    ProgramDescriptor desc;

    const auto bytes_per_element = input.element_size();
    const auto elements_per_tensor_row = input.logical_shape()[-1];
    uint32_t num_input_pages_in_row = 1;
    uint32_t num_output_pages_in_row = 1;
    uint32_t elements_per_output_page = output.logical_shape()[-1];
    uint32_t elements_per_input_page = input.logical_shape()[-1];

    if (input.is_sharded() && input.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t input_shard_width =
            (input.shard_spec().has_value() ? input.shard_spec().value().shape[1]
                                            : input.nd_shard_spec().value().shard_shape[-1]);
        num_input_pages_in_row = tt::div_up(elements_per_tensor_row, input_shard_width);
        elements_per_input_page = input_shard_width;
    }
    if (output.is_sharded() && output.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t output_shard_width =
            (output.shard_spec().has_value() ? output.shard_spec().value().shape[1]
                                             : output.nd_shard_spec().value().shard_shape[-1]);
        num_output_pages_in_row = tt::div_up(elements_per_tensor_row, output_shard_width);
        elements_per_output_page = output_shard_width;
    }

    const auto input_pages_cb_index = tt::CBIndex::c_0;
    const auto output_page_cb_index = tt::CBIndex::c_1;

    auto* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    const uint32_t total_logical_rows = input.logical_volume() / input.logical_shape()[-1];
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_logical_rows);
    std::vector<CoreCoord> ordered_cores = corerange_to_cores(all_cores, num_cores, true);

    constexpr uint32_t MAX_SUBBLOCK_SIZE_BYTES = 65536 * 4;  // Chosen empirically to prevent large row OOM CB error
    uint32_t input_page_size = input.buffer()->page_size();
    uint32_t aligned_output_page_size =
        output.buffer()->aligned_page_size();  // Since we are double buffering, the output page_size must be aligned so
    // the noc_write reads from an aligned address in the CB
    uint32_t input_subblock_size_bytes = elements_per_input_page * bytes_per_element;
    uint32_t output_subblock_size_bytes =
        elements_per_output_page *
        bytes_per_element;  // If the input/output row size is not too large, we can just set the subblock to be the
    // page and reduce the number of NoC reads/writes from/to pages.

    if (input_page_size >
        MAX_SUBBLOCK_SIZE_BYTES) {  // If the input/output row size is too large, the page size will be too large for
        // the CB, so we process data in subblock units of MAX_SUBBLOCK_SIZE_BYTES instead
        input_page_size = MAX_SUBBLOCK_SIZE_BYTES;
        input_subblock_size_bytes = MAX_SUBBLOCK_SIZE_BYTES;
    }
    if (aligned_output_page_size > MAX_SUBBLOCK_SIZE_BYTES) {
        aligned_output_page_size = MAX_SUBBLOCK_SIZE_BYTES;
        output_subblock_size_bytes = MAX_SUBBLOCK_SIZE_BYTES;
    }

    // Configuring the CB that store input pages

    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_pages_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_page_size,
        }}},
    });

    // Configuring the CB that stores output pages. This one is double buffered, since it is shared between the reader
    // and writer kernels.
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * aligned_output_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_page_cb_index),
            .data_format = output_cb_data_format,
            .page_size = aligned_output_page_size,
        }}},
    });

    // Reader kernel config with compile-time args
    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {
        static_cast<uint32_t>(input_pages_cb_index),
        static_cast<uint32_t>(output_page_cb_index),
        static_cast<uint32_t>(num_output_pages_in_row),
        static_cast<uint32_t>(num_input_pages_in_row),
        static_cast<uint32_t>(elements_per_output_page),
        static_cast<uint32_t>(bytes_per_element),
        static_cast<uint32_t>(elements_per_input_page),
        static_cast<uint32_t>(elements_per_tensor_row),
        input_subblock_size_bytes,
        output_subblock_size_bytes,
    };
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);

    // Writer kernel config with compile-time args
    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {
        static_cast<uint32_t>(output_page_cb_index),
        static_cast<uint32_t>(num_output_pages_in_row),
        static_cast<uint32_t>(elements_per_output_page),
        static_cast<uint32_t>(bytes_per_element),
        static_cast<uint32_t>(elements_per_tensor_row),
        output_subblock_size_bytes,
    };

    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = KERNEL_READER;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = KERNEL_WRITER;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Set runtime args
    uint32_t start_row_id = 0;
    for (const auto& core : ordered_cores) {
        uint32_t num_rows_to_process = num_rows_per_core_group_1;
        if (core_group_2.contains(core)) {
            num_rows_to_process = num_rows_per_core_group_2;
        }
        // Set run-time arg
        reader_desc.emplace_runtime_args(core, {input.buffer(), start_row_id, num_rows_to_process});
        writer_desc.emplace_runtime_args(core, {output.buffer(), start_row_id, num_rows_to_process});

        start_row_id += num_rows_to_process;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
