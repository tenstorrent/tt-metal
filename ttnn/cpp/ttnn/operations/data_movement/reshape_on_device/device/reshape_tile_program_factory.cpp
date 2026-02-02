// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_tile_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::prim {

ReshapeTileProgramFactory::cached_program_t ReshapeTileProgramFactory::create(
    const ttnn::prim::ReshapeOnDeviceParams& /*operation_attributes*/,
    const ttnn::prim::ReshapeOnDeviceInputs& tensor_args,
    tt::tt_metal::Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    tt::tt_metal::Buffer* src0_buffer = input_tensor.buffer();

    uint32_t num_tiles = input_tensor.physical_volume() / tt::constants::TILE_HW;

    auto output_shape = output_tensor.padded_shape();

    tt::tt_metal::Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    uint32_t alignment = src0_is_dram ? tt::tt_metal::hal::get_dram_alignment() : tt::tt_metal::hal::get_l1_alignment();

    std::vector<uint32_t> reader_compile_time_args = {alignment};
    tt::tt_metal::TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)src0_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    if (alignment > (tt::constants::FACE_WIDTH * input_tensor.element_size())) {
        uint32_t src1_cb_index = 1;
        tt::tt_metal::CircularBufferConfig cb_src1_config =
            tt::tt_metal::CircularBufferConfig(alignment, {{src1_cb_index, cb_data_format}})
                .set_page_size(src1_cb_index, alignment);
        tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);
    }

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
        "reader_unary_reshape_interleaved.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt::tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {src0_buffer->address(),
         input_tensor.padded_shape()[3] / tt::constants::TILE_WIDTH,
         (uint32_t)output_shape[0],
         (uint32_t)output_shape[1],
         (uint32_t)output_shape[2] / tt::constants::TILE_HEIGHT,
         (uint32_t)output_shape[3] / tt::constants::TILE_WIDTH});

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles, 0});

    return {std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id}};
}

void ReshapeTileProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::ReshapeOnDeviceParams& /*operation_attributes*/,
    const ttnn::prim::ReshapeOnDeviceInputs& tensor_args,
    tt::tt_metal::Tensor& output_tensor) {
    auto* src_buffer = tensor_args.input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    CoreCoord core = {0, 0};

    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    {
        auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
    }

    {
        auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
        runtime_args[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::prim
