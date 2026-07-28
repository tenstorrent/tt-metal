// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_program_factory.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

ProgramDescriptor PadTileCoreProgramFactory::create_descriptor(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;

    const CoreRangeSet core_ranges{CoreRange{{0, 0}, {0, 0}}};

    // This should allocate a DRAM buffer on the device

    const auto& output_shape = output_padded_shape;

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "pad_tile");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);
    log_debug(tt::LogOp, "output_tensor_shape: {}", output_padded_shape);
    log_debug(tt::LogOp, "input_tensor_start: {}", operation_attributes.input_tensor_start);
    log_debug(tt::LogOp, "pad_value: {}", pad_value);

    ProgramDescriptor desc;

    const uint32_t src0_cb_index = 0;
    const uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    const uint32_t src1_cb_index = 1;  // For pad buffer
    const uint32_t num_pad_tiles = 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_pad_tiles * single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
    }

    uint32_t num_unpadded_Xt = a.padded_shape()[3] / TILE_WIDTH;
    uint32_t num_total_Xt = output_shape[3] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = a.padded_shape()[2] / TILE_HEIGHT;
    uint32_t num_total_Yt = output_shape[2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;
    uint32_t num_unpadded_Z = a.padded_shape()[1];
    uint32_t num_total_Z = output_shape[1];
    uint32_t num_padded_Zt = (num_total_Z - num_unpadded_Z) * num_total_Yt * num_total_Xt;
    uint32_t num_unpadded_W = a.padded_shape()[0];
    uint32_t num_total_W = output_shape[0];
    uint32_t num_padded_Wt = (num_total_W - num_unpadded_W) * num_total_Z * num_total_Yt * num_total_Xt;

    uint32_t num_unpadded_tiles = a.physical_volume() / TILE_HW;

    // Reader compile-time args
    // Data is 32 byte aligned
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index, src1_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Tilized reader
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_ranges;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_ranges;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Slot 0 of both kernels is a raw buffer base address (no offset).  Use Buffer*
    // for BufferBinding so the framework patches addresses on cache hits without
    // rebuilding the descriptor.
    reader_desc.emplace_runtime_args(CoreCoord{0, 0}, {src0_buffer, num_unpadded_tiles, std::uint32_t{0}});

    writer_desc.emplace_runtime_args(
        CoreCoord{0, 0},
        {dst_buffer,
         num_unpadded_W,
         num_padded_Wt,
         num_unpadded_Z,
         num_padded_Zt,
         num_unpadded_Yt,
         num_padded_Yt,
         num_unpadded_Xt,
         num_padded_Xt,
         packed_pad_value});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
