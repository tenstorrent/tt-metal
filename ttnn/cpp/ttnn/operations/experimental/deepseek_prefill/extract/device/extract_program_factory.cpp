// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "extract_program_factory.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::extract {

ExtractProgramFactory::cached_program_t ExtractProgramFactory::create(
    const ExtractParams& operation_attributes, const ExtractInputs& tensor_args, Tensor& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& global_tensor = tensor_args.global_tensor;
    const auto& start = tensor_args.start;
    const auto& counts = tensor_args.counts;
    Tensor& output_tensor = tensor_return_value;

    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    const auto hidden_dim = global_tensor.logical_shape()[-1];
    const auto global_rows = global_tensor.logical_shape()[0];
    const uint32_t tiles_per_row = hidden_dim / tile_width;
    // Total tiles available in global_tensor — passed to the reader so it can
    // assert (at runtime) that start[id] + ceil_tile(counts[id]) stays within
    // the tensor bounds.
    const uint32_t global_num_tiles = (global_rows / tt::constants::TILE_HEIGHT) * tiles_per_row;
    // Maximum tiles we're allowed to emit into the output — passed to both
    // kernels so they can assert ceil_tile(counts[id]) <= max_dispatched_tokens_per_expert.
    const uint32_t max_output_tiles =
        (operation_attributes.max_dispatched_tokens_per_expert / tt::constants::TILE_HEIGHT) * tiles_per_row;

    // NOTE: We do NOT verify the inter-expert layout invariant
    //   start[id] + counts[id] <= start[id + 1]
    // i.e. that one expert's slice doesn't bleed into the next expert's slice
    // in global_tensor. That's the caller's contract to uphold. Enforcing it
    // would require reading start/counts host-side (which we explicitly avoid
    // to keep the op device-local across a mesh) or cross-iteration state in
    // the kernels. The bounds checks below cover only "stay inside global_tensor"
    // and "stay inside the output tensor".

    auto* global_buffer = global_tensor.buffer();
    auto* start_buffer = start.buffer();
    auto* counts_buffer = counts.buffer();
    auto* output_buffer = output_tensor.buffer();

    const tt::DataFormat tile_data_format = tt::tt_metal::datatype_to_dataformat_converter(global_tensor.dtype());
    const uint32_t single_tile_size = tt::tile_size(tile_data_format);

    const uint32_t start_page_size = start_buffer->aligned_page_size();
    const uint32_t counts_page_size = counts_buffer->aligned_page_size();
    const tt::DataFormat idx_data_format = tt::tt_metal::datatype_to_dataformat_converter(start.dtype());

    // Single-core implementation: reader streams tiles, writer drains them.
    const CoreCoord core{0, 0};
    const CoreRange core_range{core, core};
    const CoreRangeSet core_range_set{core_range};

    constexpr uint32_t cb_tile = tt::CBIndex::c_0;
    constexpr uint32_t cb_start_scratch = tt::CBIndex::c_1;
    constexpr uint32_t cb_counts_scratch_reader = tt::CBIndex::c_2;
    constexpr uint32_t cb_counts_scratch_writer = tt::CBIndex::c_3;

    // Deep pipeline so the reader can race ahead of the writer (and vice versa)
    // without stalling on cb_reserve_back / cb_wait_front. Each tile is small
    // (e.g. ~1 KB for bfp8_b) so even 32 slots is only ~32 KB of L1.
    constexpr uint32_t tile_buffering = 32;
    tt::tt_metal::CircularBufferConfig cb_tile_config =
        tt::tt_metal::CircularBufferConfig(tile_buffering * single_tile_size, {{cb_tile, tile_data_format}})
            .set_page_size(cb_tile, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_tile_config);

    tt::tt_metal::CircularBufferConfig cb_start_config =
        tt::tt_metal::CircularBufferConfig(start_page_size, {{cb_start_scratch, idx_data_format}})
            .set_page_size(cb_start_scratch, start_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_start_config);

    tt::tt_metal::CircularBufferConfig cb_counts_reader_config =
        tt::tt_metal::CircularBufferConfig(counts_page_size, {{cb_counts_scratch_reader, idx_data_format}})
            .set_page_size(cb_counts_scratch_reader, counts_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_counts_reader_config);

    tt::tt_metal::CircularBufferConfig cb_counts_writer_config =
        tt::tt_metal::CircularBufferConfig(counts_page_size, {{cb_counts_scratch_writer, idx_data_format}})
            .set_page_size(cb_counts_scratch_writer, counts_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_counts_writer_config);

    // Reader compile-time args: CB ids, scalars, then TensorAccessorArgs for global/start/counts.
    std::vector<uint32_t> reader_compile_time_args = {
        cb_tile,
        cb_start_scratch,
        cb_counts_scratch_reader,
        operation_attributes.global_expert_id,
        tiles_per_row,
        global_num_tiles,
        max_output_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(global_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(start_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(counts_buffer).append_to(reader_compile_time_args);

    // Writer compile-time args: CB ids, scalars, then TensorAccessorArgs for output/counts.
    std::vector<uint32_t> writer_compile_time_args = {
        cb_tile,
        cb_counts_scratch_writer,
        operation_attributes.global_expert_id,
        tiles_per_row,
        max_output_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(counts_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/extract/device/kernels/dataflow/reader_extract.cpp",
        core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/extract/device/kernels/dataflow/writer_extract.cpp",
        core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt::tt_metal::SetRuntimeArgs(
        program, reader_kernel_id, core, {global_buffer->address(), start_buffer->address(), counts_buffer->address()});
    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, {output_buffer->address(), counts_buffer->address()});

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id, .writer_kernel_id = writer_kernel_id, .cores = {core}}};
}

void ExtractProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ExtractParams& /*operation_attributes*/,
    const ExtractInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    const uint32_t global_addr = tensor_args.global_tensor.buffer()->address();
    const uint32_t start_addr = tensor_args.start.buffer()->address();
    const uint32_t counts_addr = tensor_args.counts.buffer()->address();
    const uint32_t output_addr = tensor_return_value.buffer()->address();

    for (const auto& core : cores) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
        reader_args[0] = global_addr;
        reader_args[1] = start_addr;
        reader_args[2] = counts_addr;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
        writer_args[0] = output_addr;
        writer_args[1] = counts_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::extract
