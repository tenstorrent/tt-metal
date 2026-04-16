// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::index_fill {

using namespace tt::tt_metal;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/index_fill/device/kernels/reader_index_fill.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/index_fill/device/kernels/writer_index_fill.cpp";

ProgramDescriptor IndexFillOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const Tensor& index = tensor_args.index;
    const Tensor& input = tensor_args.input;
    uint32_t dim = operation_attributes.dim;

    const auto input_shape = input.padded_shape();
    const auto n = input_shape.rank();
    uint32_t num_rows_in_dim = 1;
    for (int i = n - 2; i > static_cast<int>(dim); --i) {
        num_rows_in_dim *= input_shape[i];
    }

    // Prepare fill_value to send as a uint32_t kernel arg
    auto fill_value_ = operation_attributes.value;
    uint32_t fill_value{};
    switch (input.dtype()) {
        case DataType::BFLOAT16:
            fill_value = pack_two_bfloat16_into_uint32({bfloat16(std::get<float>(fill_value_)), bfloat16(0.0f)});
            break;
        case DataType::FLOAT32: fill_value = std::bit_cast<uint32_t>(std::get<float>(fill_value_)); break;
        case DataType::INT32: fill_value = static_cast<uint32_t>(std::get<int>(fill_value_)); break;
        default: TT_FATAL(false, "Unsupported datatype"); break;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                            Program Setup
    ////////////////////////////////////////////////////////////////////////////
    // Distribute work across core grid
    auto num_rows = input.physical_volume() / input.padded_shape()[-1];
    auto compute_with_storage_grid_size = input.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_rows);

    // Create circular buffers
    auto input_dataformat = datatype_to_dataformat_converter(input.dtype());
    auto index_dataformat = datatype_to_dataformat_converter(index.dtype());

    uint32_t input_page_size = input.buffer()->aligned_page_size();
    uint32_t index_total_size = index.buffer()->aligned_size();
    uint32_t output_page_size = output.buffer()->aligned_page_size();

    uint32_t input_cb_depth = 2;

    ProgramDescriptor desc;

    // CB to store pages from input tensor
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_depth * input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = input_dataformat,
            .page_size = input_page_size,
        }}},
    });

    // CB to store entire index tensor
    desc.cbs.push_back(CBDescriptor{
        .total_size = index_total_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = index_dataformat,
            .page_size = index_total_size,
        }}},
    });

    // CB to store an input page filled with fill_value
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2,
            .data_format = input_dataformat,
            .page_size = input_page_size,
        }}},
    });

    // Reader kernel
    KernelDescriptor::CompileTimeArgs reader_ct_args = {
        input_page_size,                                 // input tensor page size
        index_total_size,                                // index tensor total size
        static_cast<uint32_t>(index.physical_volume()),  // num elements in index array
        static_cast<uint32_t>(dim == n - 1),             // is last dim
    };
    TensorAccessorArgs(input.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(index.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    KernelDescriptor::CompileTimeArgs writer_ct_args = {
        output_page_size,                                // output tensor page size
        static_cast<uint32_t>(index.physical_volume()),  // num elements in index array
        static_cast<uint32_t>(input.element_size()),     // element size in bytes
        static_cast<uint32_t>(dim == n - 1),             // is last dim
    };
    TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = std::move(all_cores);
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Set runtime args for each core
    uint32_t start_row_id = 0;
    auto cores = corerange_to_cores(all_cores);
    reader_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());

    for (const auto& core : cores) {
        uint32_t num_rows_per_core{};
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                input.buffer()->address(),         // input tensor address
                index.buffer()->address(),         // index tensor address
                start_row_id,                      // start row
                start_row_id + num_rows_per_core,  // end row
                num_rows_in_dim,                   // num rows in dim
                input_shape[dim]});                // dim size

        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                output.buffer()->address(),        // output tensor address
                start_row_id,                      // start row
                start_row_id + num_rows_per_core,  // end row
                num_rows_in_dim,                   // num rows in dim
                input_shape[dim],                  // dim size
                fill_value});                      // fill value

        start_row_id += num_rows_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::index_fill
