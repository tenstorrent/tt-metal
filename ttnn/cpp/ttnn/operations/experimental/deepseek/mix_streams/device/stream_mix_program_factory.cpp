// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "stream_mix_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek::mix_streams {

namespace {

// Stream (hc) and sublayer_out tiles of one work unit, consumed together.
constexpr uint32_t kStreamMixCbIn = tt::CBIndex::c_0;
// The row block's diagonal weight tiles: hc * n_out mixing tiles, then n_out post tiles.
constexpr uint32_t kStreamMixCbW = tt::CBIndex::c_1;
// Staging for face 0 of each of a row block's 32 per-token weight / post tiles.
constexpr uint32_t kStreamMixCbWSrc = tt::CBIndex::c_2;
constexpr uint32_t kStreamMixCbPostSrc = tt::CBIndex::c_3;
constexpr uint32_t kStreamMixCbOut = tt::CBIndex::c_4;

// 32 tokens x one 16x16 bf16 face (512 B) = 16 KB = 8 tiles.
constexpr uint32_t kStreamMixStagePages = 8;

constexpr char kStreamMixReaderPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/device/kernels/dataflow/reader_stream_mix.cpp";
constexpr char kStreamMixComputePath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/device/kernels/compute/stream_mix.cpp";
constexpr char kStreamMixWriterPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/device/kernels/dataflow/writer_stream_mix.cpp";

}  // namespace

StreamMixProgramFactory::cached_program_t StreamMixProgramFactory::create(
    const StreamMixParams& operation_attributes,
    const StreamMixInputs& tensor_args,
    StreamMixTensorReturn& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& streams = tensor_args.streams;
    const auto& weights = tensor_args.weights;
    const bool has_post = tensor_args.post.has_value();
    const Buffer* post_buffer = has_post ? tensor_args.post->buffer() : nullptr;
    const Buffer* sub_buffer = has_post ? tensor_args.sublayer_out->buffer() : nullptr;
    auto& output = tensor_return_value;

    const uint32_t hc = operation_attributes.num_streams;
    const uint32_t n_out = operation_attributes.num_outputs;
    const uint32_t num_tokens = static_cast<uint32_t>(streams.logical_shape()[2]);
    const uint32_t t_tiles = streams.padded_shape()[2] / constants::TILE_HEIGHT;
    const uint32_t d_tiles = streams.padded_shape()[3] / constants::TILE_WIDTH;
    const uint32_t total_units = t_tiles * d_tiles;
    const uint32_t n_in = hc + (has_post ? 1u : 0u);
    const uint32_t n_w = (hc + (has_post ? 1u : 0u)) * n_out;

    Program program = CreateProgram();

    const DataFormat tile_data_format = datatype_to_dataformat_converter(streams.dtype());
    const uint32_t tile_size_bytes = tile_size(tile_data_format);

    IDevice* device = streams.device();
    const CoreCoord grid_size = device->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2] =
        split_work_to_cores(grid_size, total_units, /*row_wise=*/true);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    constexpr uint32_t tile_buffering = 2;
    auto make_cb = [&](uint32_t index, uint32_t num_pages) {
        CircularBufferConfig config = CircularBufferConfig(num_pages * tile_size_bytes, {{index, tile_data_format}})
                                          .set_page_size(index, tile_size_bytes);
        CreateCircularBuffer(program, all_cores, config);
    };
    make_cb(kStreamMixCbIn, tile_buffering * n_in);
    make_cb(kStreamMixCbW, n_w);
    make_cb(kStreamMixCbWSrc, kStreamMixStagePages);
    if (has_post) {
        make_cb(kStreamMixCbPostSrc, kStreamMixStagePages);
    }
    make_cb(kStreamMixCbOut, tile_buffering * n_out);

    std::vector<uint32_t> reader_compile_time_args = {
        kStreamMixCbIn,
        kStreamMixCbW,
        kStreamMixCbWSrc,
        kStreamMixCbPostSrc,
        hc,
        n_out,
        has_post ? 1u : 0u,
        d_tiles,
        t_tiles,
        num_tokens,
    };
    TensorAccessorArgs(streams.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(weights.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(post_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(sub_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {kStreamMixCbOut, n_out, d_tiles, t_tiles};
    TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    const std::vector<uint32_t> compute_compile_time_args = {
        kStreamMixCbW,
        kStreamMixCbIn,
        kStreamMixCbOut,
        hc,
        n_out,
        has_post ? 1u : 0u,
        d_tiles,
    };

    const KernelHandle reader_kernel_id =
        CreateKernel(program, kStreamMixReaderPath, all_cores, ReaderDataMovementConfig(reader_compile_time_args));
    const KernelHandle writer_kernel_id =
        CreateKernel(program, kStreamMixWriterPath, all_cores, WriterDataMovementConfig(writer_compile_time_args));
    const KernelHandle compute_kernel_id = CreateKernel(
        program,
        kStreamMixComputePath,
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args});

    const uint32_t post_addr = has_post ? post_buffer->address() : 0u;
    const uint32_t sub_addr = has_post ? sub_buffer->address() : 0u;
    const std::vector<CoreCoord> cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    uint32_t start_unit = 0;
    for (const auto& core : cores) {
        uint32_t units_this_core = 0;
        if (core_group_1.contains(core)) {
            units_this_core = units_per_core_1;
        } else if (core_group_2.contains(core)) {
            units_this_core = units_per_core_2;
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {streams.buffer()->address(),
             weights.buffer()->address(),
             post_addr,
             sub_addr,
             start_unit,
             units_this_core});
        SetRuntimeArgs(program, compute_kernel_id, core, {start_unit, units_this_core});
        SetRuntimeArgs(program, writer_kernel_id, core, {output.buffer()->address(), start_unit, units_this_core});

        start_unit += units_this_core;
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id, .writer_kernel_id = writer_kernel_id, .cores = cores}};
}

void StreamMixProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const StreamMixParams& /*operation_attributes*/,
    const StreamMixInputs& tensor_args,
    StreamMixTensorReturn& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const bool has_post = tensor_args.post.has_value();
    const uint32_t streams_addr = tensor_args.streams.buffer()->address();
    const uint32_t weights_addr = tensor_args.weights.buffer()->address();
    const uint32_t post_addr = has_post ? tensor_args.post->buffer()->address() : 0u;
    const uint32_t sub_addr = has_post ? tensor_args.sublayer_out->buffer()->address() : 0u;
    const uint32_t out_addr = tensor_return_value.buffer()->address();

    auto& reader_args_by_core = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& reader_args = reader_args_by_core[core.x][core.y];
        reader_args[0] = streams_addr;
        reader_args[1] = weights_addr;
        reader_args[2] = post_addr;
        reader_args[3] = sub_addr;

        writer_args_by_core[core.x][core.y][0] = out_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek::mix_streams
