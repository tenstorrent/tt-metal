// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_tile_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/math.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor ReshapeTileProgramFactory::create_descriptor(
    const ttnn::prim::ReshapeOnDeviceParams& /*operation_attributes*/,
    const ttnn::prim::ReshapeOnDeviceInputs& tensor_args,
    tt::tt_metal::Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;

    const CoreRangeSet core_ranges{CoreRange{{0, 0}, {0, 0}}};
    const CoreCoord core{0, 0};

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    Buffer* src0_buffer = input_tensor.buffer();

    uint32_t num_tiles = input_tensor.physical_volume() / tt::constants::TILE_HW;

    auto output_shape = output_tensor.padded_shape();

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    constexpr uint32_t src0_cb_index = 0;
    constexpr uint32_t num_input_tiles = 2;

    ProgramDescriptor desc;

    // Primary input CB: holds tiles streamed from DRAM/L1 into the writer.
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM;
    uint32_t alignment = src0_is_dram ? hal::get_dram_alignment() : hal::get_l1_alignment();

    std::vector<uint32_t> reader_compile_time_args = {alignment};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(src0_cb_index)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Wide-alignment scratch CB: only needed when DRAM/L1 alignment exceeds a face's
    // worth of bytes (the reader uses it as a temporary aligned landing pad).
    if (alignment > (tt::constants::FACE_WIDTH * input_tensor.element_size())) {
        constexpr uint32_t src1_cb_index = 1;
        desc.cbs.push_back(CBDescriptor{
            .total_size = alignment,
            .core_ranges = core_ranges,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src1_cb_index),
                .data_format = cb_data_format,
                .page_size = alignment,
            }}},
        });
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
        "reader_unary_reshape_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_ranges;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Reader runtime arg slot 0: src buffer address. Buffer* registers a BufferBinding
    // so the framework patches the address on cache hits without rebuilding the descriptor.
    reader_desc.emplace_runtime_args(
        core,
        {src0_buffer,
         input_tensor.padded_shape()[3] / tt::constants::TILE_WIDTH,
         static_cast<uint32_t>(output_shape[0]),
         static_cast<uint32_t>(output_shape[1]),
         static_cast<uint32_t>(output_shape[2]) / tt::constants::TILE_HEIGHT,
         static_cast<uint32_t>(output_shape[3]) / tt::constants::TILE_WIDTH});

    desc.kernels.push_back(std::move(reader_desc));

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_ranges;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Writer runtime arg slot 0: dst buffer address (Buffer* -> BufferBinding for cache-hit patching).
    // Slot 1: total tile count, slot 2: start_id (always 0 here, single-core).
    writer_desc.emplace_runtime_args(core, {dst_buffer, num_tiles, 0u});

    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
