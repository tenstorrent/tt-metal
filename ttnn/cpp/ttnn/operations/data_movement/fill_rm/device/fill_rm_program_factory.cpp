// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor FillRMProgramFactory::create_descriptor(
    const FillRmParams& operation_attributes, const FillRmInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    const uint32_t N = operation_attributes.N;
    const uint32_t C = operation_attributes.C;
    const uint32_t H = operation_attributes.H;
    const uint32_t W = operation_attributes.W;
    const uint32_t hFill = operation_attributes.hFill;
    const uint32_t wFill = operation_attributes.wFill;
    const float val_hi = operation_attributes.val_hi;
    const float val_lo = operation_attributes.val_lo;

    const CoreRangeSet core_ranges{CoreRange{{0, 0}, {0, 0}}};

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);

    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t num_cb_tiles = 16;
    TT_FATAL(
        W < 1024 * num_cb_tiles,
        "Width (W) must be less than {} for kernel simplification. Got W={}, num_cb_tiles={}",
        1024 * num_cb_tiles,
        W,
        num_cb_tiles);

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_tiles * single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_tiles * single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 1,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*dst_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/fill_rm/device/kernels/dataflow/fill_rm_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_ranges;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Buffer* triggers a binding entry for dst_buffer at arg slot 0; the framework
    // patches its address on cache hits without rebuilding the descriptor.
    reader_desc.emplace_runtime_args(
        CoreCoord{0, 0},
        {dst_buffer,
         uint32_t(N * C),
         uint32_t(H),
         uint32_t(W),
         uint32_t(hFill),
         uint32_t(wFill),
         uint32_t(std::bit_cast<uint16_t>(bfloat16(val_hi))),
         uint32_t(std::bit_cast<uint16_t>(bfloat16(val_lo)))});

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::prim
