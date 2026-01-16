// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_program_factory.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/host_api.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::pad::program {
PadTileCoreProgramFactory::cached_program_t PadTileCoreProgramFactory::create(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    tt::tt_metal::Program program{};

    CoreRange core({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device

    auto output_shape = output_padded_shape;

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "pad_tile");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);
    log_debug(tt::LogOp, "output_tensor_shape: {}", output_padded_shape);
    log_debug(tt::LogOp, "input_tensor_start: {}", operation_attributes.input_tensor_start);
    log_debug(tt::LogOp, "pad_value: {}", pad_value);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;  // For pad buffer
    uint32_t num_pad_tiles = 1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(num_pad_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

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

    const std::array reader_kernel_args = {
        src0_buffer->address(),
        num_unpadded_tiles,
        std::uint32_t{0},
    };
    const std::array writer_kernel_args = {
        dst_buffer->address(),
        num_unpadded_W,
        num_padded_Wt,
        num_unpadded_Z,
        num_padded_Zt,
        num_unpadded_Yt,
        num_padded_Yt,
        num_unpadded_Xt,
        num_padded_Xt,
        packed_pad_value,
    };

    // Reader compile-time args
    // Data is 32 byte aligned
    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index, src1_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);

    return cached_program_t{std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id}};
}

void PadTileCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PadParams& /*operation_attributes*/,
    const PadInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* src_dram_buffer = tensor_args.input.buffer();
    auto* dst_dram_buffer = tensor_return_value.buffer();

    CoreCoord core = {0, 0};

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(
            cached_program.program, cached_program.shared_variables.unary_reader_kernel_id, core);
        runtime_args[0] = src_dram_buffer->address();
    }

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(
            cached_program.program, cached_program.shared_variables.unary_writer_kernel_id, core);
        runtime_args[0] = dst_dram_buffer->address();
    }
}

}  // namespace ttnn::operations::data_movement::pad::program
