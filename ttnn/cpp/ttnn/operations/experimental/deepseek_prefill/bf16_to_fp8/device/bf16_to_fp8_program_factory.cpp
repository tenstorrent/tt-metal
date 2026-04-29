// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bf16_to_fp8_program_factory.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8 {

Bf16ToFp8ProgramFactory::cached_program_t Bf16ToFp8ProgramFactory::create(
    const Bf16ToFp8Params& /*operation_attributes*/, const Bf16ToFp8Inputs& tensor_args, Tensor& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& input_tensor = tensor_args.input_tensor;
    Tensor& output_tensor = tensor_return_value;

    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();

    const auto& shape = input_tensor.logical_shape();
    const uint32_t total_elems = shape.volume();
    const uint32_t tile_h = tt::constants::TILE_HEIGHT;
    const uint32_t tile_w = tt::constants::TILE_WIDTH;
    const uint32_t num_tiles = total_elems / (tile_h * tile_w);

    // Single-core: walk all tiles sequentially on (0,0).
    const CoreCoord core{0, 0};
    const CoreRangeSet core_range_set{CoreRange{core, core}};

    constexpr tt::DataFormat in_data_format = tt::DataFormat::Float16_b;
    constexpr tt::DataFormat out_data_format = tt::DataFormat::Fp8_e4m3;
    const uint32_t in_tile_bytes = tt::tile_size(in_data_format);
    const uint32_t out_tile_bytes = tt::tile_size(out_data_format);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_1;
    constexpr uint32_t cb_buffering = 2;

    tt::tt_metal::CircularBufferConfig cb_in_config =
        tt::tt_metal::CircularBufferConfig(cb_buffering * in_tile_bytes, {{cb_in, in_data_format}})
            .set_page_size(cb_in, in_tile_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_in_config);

    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(cb_buffering * out_tile_bytes, {{cb_out, out_data_format}})
            .set_page_size(cb_out, out_tile_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_out_config);

    std::vector<uint32_t> reader_compile_time_args = {cb_in, num_tiles, in_tile_bytes};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {cb_out, num_tiles, out_tile_bytes};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {cb_in, cb_out, num_tiles};

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/bf16_to_fp8/device/kernels/dataflow/"
        "reader_bf16_to_fp8.cpp",
        core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/bf16_to_fp8/device/kernels/dataflow/"
        "writer_bf16_to_fp8.cpp",
        core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/bf16_to_fp8/device/kernels/compute/"
        "bf16_to_fp8_compute.cpp",
        core_range_set,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, {input_buffer->address()});
    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, {output_buffer->address()});

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .cores = {core}}};
}

void Bf16ToFp8ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const Bf16ToFp8Params& /*operation_attributes*/,
    const Bf16ToFp8Inputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t output_addr = tensor_return_value.buffer()->address();

    for (const auto& core : cores) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
        reader_args[0] = input_addr;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
        writer_args[0] = output_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8
