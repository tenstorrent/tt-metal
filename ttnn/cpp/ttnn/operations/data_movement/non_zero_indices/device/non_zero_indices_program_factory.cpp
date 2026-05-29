// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor NonZeroIndicesProgramFactory::create_descriptor(
    const NonzeroParams& /*operation_attributes*/, const NonzeroInputs& tensor_args, NonzeroResult& output_tensors) {
    const auto& input = tensor_args.input;
    const auto& out_num_indices = std::get<0>(output_tensors);
    const auto& out_indices = std::get<1>(output_tensors);

    // we want per core to be aligned to aligment_base per core
    const uint32_t alignment_base = 32 / input.element_size();
    const uint32_t aligned_elements = tt::div_up(input.padded_shape()[-1], alignment_base) * alignment_base;
    const uint32_t actual_elements = input.padded_shape()[-1];

    const CoreCoord core = {0, 0};
    const CoreRangeSet core_ranges{CoreRange{core, core}};

    constexpr uint32_t input_cb_index = 0;
    constexpr uint32_t output_cb_index_0 = 1;
    constexpr uint32_t output_cb_index_1 = 2;

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(DataType::UINT32);

    const uint32_t page_size = actual_elements * input.element_size();
    const uint32_t rounded_page_size = round_up_to_mul32(page_size);

    const uint32_t dst_page_size = actual_elements * 4;
    const uint32_t dst_rounded_page_size = round_up_to_mul32(dst_page_size);

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * rounded_page_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = input_cb_index,
            .data_format = input_cb_data_format,
            .page_size = rounded_page_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * 32,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index_0,
            .data_format = output_cb_data_format,
            .page_size = 32,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * dst_rounded_page_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index_1,
            .data_format = output_cb_data_format,
            .page_size = dst_rounded_page_size,
        }}},
    });

    Buffer* const src_buffer = input.buffer();
    Buffer* const out_num_indices_buffer = out_num_indices.buffer();
    Buffer* const out_indices_buffer = out_indices.buffer();

    std::vector<uint32_t> compile_time_args = {
        input_cb_index,
        output_cb_index_0,
        output_cb_index_1,
    };
    TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*out_num_indices_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*out_indices_buffer).append_to(compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/non_zero_indices/device/kernels/dataflow/"
        "non_zero_indices_sc_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_ranges;
    reader_desc.compile_time_args = std::move(compile_time_args);
    reader_desc.defines = {{"NUM_BYTES", std::to_string(input.element_size())}};
    reader_desc.config = ReaderConfigDescriptor{};

    // Buffer* entries register as buffer bindings at their arg slots; the framework
    // patches their addresses on cache hits without rebuilding the descriptor.
    reader_desc.emplace_runtime_args(
        core,
        {src_buffer,
         out_num_indices_buffer,
         out_indices_buffer,
         aligned_elements,
         actual_elements,
         uint32_t(input.element_size())});

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::prim
