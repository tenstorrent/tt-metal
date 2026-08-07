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

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_units = W;
    auto* src_buffer = input.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const uint32_t page_alignment =
        src_is_dram ? tt::tt_metal::hal::get_dram_alignment() : tt::tt_metal::hal::get_l1_alignment();
    uint32_t aligned_input_page_size = tt::align(num_input_units * input_unit_size, page_alignment);

    // When the input is sharded, bind the CB to the input buffer so the framework
    // can re-apply the globally-allocated address on a program-cache hit. For the
    // interleaved path the CB is plain L1 scratch and the buffer address is passed
    // through the reader runtime arg instead.
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
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
    reader_desc.config = ReaderConfigDescriptor{};

    auto cores = corerange_to_cores(all_cores, num_cores, true);

    // Pass Buffer* (not address) so the framework's cache-hit path can re-patch
    // the runtime arg without rebuilding the descriptor.
    for (const auto& core : cores) {
        reader_desc.emplace_runtime_args(core, {src_buffer});
    }

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
