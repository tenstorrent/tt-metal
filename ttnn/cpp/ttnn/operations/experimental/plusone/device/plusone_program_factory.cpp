// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>

#include "plusone_program_factory.hpp"
#include "plusone_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor PlusOneProgramFactory::create_descriptor(
    const PlusoneParams& operation_attributes, const Tensor& input, Tensor& /*tensor_return_value*/) {
    ProgramDescriptor desc;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = input.element_size();

    CoreRangeSet all_cores = CoreRangeSet(std::vector{CoreRange({0, 0}, {0, 0})});
    uint32_t num_cores = 1;  // single-core

    if (operation_attributes.sub_core_grids.has_value()) {
        all_cores = operation_attributes.sub_core_grids.value();
        num_cores = all_cores.num_cores();
    }

    const auto& input_shape = input.padded_shape();
    uint32_t W = input_shape[-1];
    uint32_t H = 1;
    if (!input.is_sharded() && input_shape.size() > 1) {
        for (uint32_t i = 0; i < input_shape.size() - 1; ++i) {
            H *= input_shape[i];
        }
    }

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t num_input_units = W;
    auto* src_buffer = input.buffer();
    const bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const uint32_t page_alignment =
        src_is_dram ? tt::tt_metal::hal::get_dram_alignment() : tt::tt_metal::hal::get_l1_alignment();
    const uint32_t aligned_input_page_size = tt::align(num_input_units * input_unit_size, page_alignment);

    // When the input is sharded, bind the CB to the input buffer for dynamic CB
    // address re-application on cache hit; otherwise the CB is plain L1 scratch
    // and the buffer address is supplied through the runtime arg.
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = input_cb_data_format,
            .page_size = aligned_input_page_size,
        }}},
        .buffer = input.is_sharded() ? src_buffer : nullptr,
    });

    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index, src_is_dram, aligned_input_page_size, W, H, operation_attributes.skip_negative_entries};
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/plusone/device/kernels/reader_plusone_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderDataMovementConfig{};

    const auto cores = corerange_to_cores(all_cores, num_cores, true);
    reader_desc.runtime_args.reserve(cores.size());
    for (const auto& core : cores) {
        reader_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{src_buffer->address()});
    }

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
