// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

RotateHalfProgramFactory::cached_program_t RotateHalfProgramFactory::create(
    const RotateHalfParams& /*operation_attributes*/, const Tensor& input, Tensor& tensor_return_value) {
    using namespace tt::constants;

    Program program{};

    const CoreCoord core({0, 0});
    CoreRange core_range(core, core);

    Tensor& output = tensor_return_value;

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);

    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    const uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);

    const uint32_t num_tiles = input.physical_volume() / TILE_HW;
    const uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT;
    const uint32_t half_row_size = input.padded_shape()[-1] / TILE_WIDTH / 2;

    // Used for half of tensor that is multiplied
    const uint32_t src_mul_cb_index = 0;
    const uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src_mul_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src_mul_cb_index, cb_data_format}})
            .set_page_size(src_mul_cb_index, single_tile_size);
    CreateCircularBuffer(program, core_range, cb_src_mul_config);

    // Used for bcast scalar
    const uint32_t src_scalar_cb_index = 1;
    const uint32_t num_scalar_tiles = 1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_scalar_tiles * scalar_single_tile_size, {{src_scalar_cb_index, cb_data_format}})
            .set_page_size(src_scalar_cb_index, scalar_single_tile_size);
    CreateCircularBuffer(program, core_range, cb_src1_config);

    // Used for half of tensor that is not multiplied
    const uint32_t src_no_mul_cb_index = 2;
    CircularBufferConfig cb_src_no_mul_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src_no_mul_cb_index, cb_data_format}})
            .set_page_size(src_no_mul_cb_index, single_tile_size);
    CreateCircularBuffer(program, core_range, cb_src_no_mul_config);

    const uint32_t output_mul_cb_index = tt::CBIndex::c_16;
    const uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_mul_cb_index, cb_data_format}})
            .set_page_size(output_mul_cb_index, single_tile_size);
    CreateCircularBuffer(program, core_range, cb_output_config);
    const uint32_t output_no_mul_cb_index = src_no_mul_cb_index;

    const uint16_t bfloat16_scalar = std::bit_cast<uint16_t>(bfloat16(-1.0f));

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        src_no_mul_cb_index, src_mul_cb_index, src_scalar_cb_index, static_cast<uint32_t>(bfloat16_scalar)};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {output_no_mul_cb_index, output_mul_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/device/kernels/dataflow/"
        "reader_rotate_half_interleaved_start_id.cpp",
        core_range,
        ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/device/kernels/dataflow/"
        "writer_rotate_half_interleaved_start_id.cpp",
        core_range,
        WriterDataMovementConfig(writer_compile_time_args));

    std::map<std::string, std::string> bcast_compute_defines = {
        {"BCAST_OP", "mul_tiles_bcast"},
        {"BCAST_LLKOP", "ELWMUL"},
        {"BCAST_DIM", "BroadcastType::SCALAR"},
        {"BCAST_SCALAR", "1"}};

    auto bcast_kernel_group_1_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw.cpp",
        core_range,
        ComputeConfig{.compile_args = {}, .defines = bcast_compute_defines});

    SetRuntimeArgs(program, reader_kernel_id, core_range, {src_buffer->address(), num_rows, half_row_size, 0});

    SetRuntimeArgs(
        program,
        bcast_kernel_group_1_id,
        core_range,
        {
            1,             // B
            1,             // Ht
            num_tiles / 2  // Wt
        });

    SetRuntimeArgs(program, writer_kernel_id, core_range, {dst_buffer->address(), num_rows, half_row_size, 0});

    return cached_program_t{
        std::move(program), {.reader_kernel_id = reader_kernel_id, .writer_kernel_id = writer_kernel_id, .core = core}};
}

void RotateHalfProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const RotateHalfParams& /*operation_attributes*/,
    const Tensor& input,
    Tensor& tensor_return_value) {
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = tensor_return_value.buffer();

    const auto& core = cached_program.shared_variables.core;

    {
        auto& runtime_args =
            GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
    }

    {
        auto& runtime_args =
            GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id, core);
        runtime_args[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::experimental::prim
