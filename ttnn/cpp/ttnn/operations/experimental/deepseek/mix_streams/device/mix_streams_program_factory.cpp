// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mix_streams_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>

namespace ttnn::operations::experimental::deepseek::mix_streams {

namespace {

constexpr uint32_t kCbCombSrc = tt::CBIndex::c_0;
constexpr uint32_t kCbComb = tt::CBIndex::c_1;
constexpr uint32_t kCbPostSrc = tt::CBIndex::c_2;
constexpr uint32_t kCbPost = tt::CBIndex::c_3;
constexpr uint32_t kCbStreams = tt::CBIndex::c_4;
constexpr uint32_t kCbSub = tt::CBIndex::c_5;
constexpr uint32_t kCbOut = tt::CBIndex::c_6;

// The per-tile work (two single-tile matmuls) is dwarfed by the per-core setup -- rebuilding
// the comb / post tiles and paying the DRAM round trip for them -- so spreading the tiny
// decode shapes over the whole grid costs more than it saves. Measured on a [1,1,4,4096]
// decode step: 18.5 us at 4 cores, 8.1 at 16, 7.3 at 32, 8.1 at 64.
constexpr uint32_t kMaxCores = 32;

constexpr char kReaderKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/device/kernels/dataflow/reader_mix_streams.cpp";
constexpr char kComputeKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/device/kernels/compute/mix_streams.cpp";
constexpr char kWriterKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/device/kernels/dataflow/writer_mix_streams.cpp";

}  // namespace

MixStreamsProgramFactory::cached_program_t MixStreamsProgramFactory::create(
    const MixStreamsParams& operation_attributes,
    const MixStreamsInputs& tensor_args,
    MixStreamsTensorReturn& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& post = tensor_args.post;
    const auto& comb = tensor_args.comb;
    const auto& sublayer_out = tensor_args.sublayer_out;
    const auto& streams = tensor_args.streams;
    auto& output = tensor_return_value;

    Program program = CreateProgram();

    const DataFormat tile_data_format = datatype_to_dataformat_converter(streams.dtype());
    const uint32_t tile_size_bytes = tile_size(tile_data_format);

    const auto& streams_shape = streams.logical_shape();
    const uint32_t hc = operation_attributes.num_streams;
    const uint32_t num_tokens = static_cast<uint32_t>(streams_shape[0]) * static_cast<uint32_t>(streams_shape[1]);
    const uint32_t n_tiles = streams.padded_shape()[-1] / constants::TILE_WIDTH;
    // hc <= TILE_HEIGHT, so every token is exactly one tile row of the [T*hc, D] tile grid.
    const uint32_t total_tiles = num_tokens * n_tiles;

    IDevice* device = streams.device();
    const CoreCoord grid_size = device->compute_with_storage_grid_size();
    const uint32_t max_cores = std::min<uint32_t>(
        total_tiles, std::min<uint32_t>(kMaxCores, static_cast<uint32_t>(grid_size.x * grid_size.y)));
    const CoreRangeSet work_grid = num_cores_to_corerangeset(max_cores, grid_size, /*row_wise=*/true);
    const auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2] =
        split_work_to_cores(work_grid, total_tiles, /*row_wise=*/true);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    constexpr uint32_t tile_buffering = 2;
    auto make_cb = [&](uint32_t index, uint32_t num_pages) {
        CircularBufferConfig config = CircularBufferConfig(num_pages * tile_size_bytes, {{index, tile_data_format}})
                                          .set_page_size(index, tile_size_bytes);
        CreateCircularBuffer(program, all_cores, config);
    };
    make_cb(kCbCombSrc, 1);
    make_cb(kCbComb, tile_buffering);
    make_cb(kCbPostSrc, 1);
    make_cb(kCbPost, tile_buffering);
    make_cb(kCbStreams, tile_buffering);
    make_cb(kCbSub, tile_buffering);
    make_cb(kCbOut, tile_buffering);

    std::vector<uint32_t> reader_compile_time_args = {
        kCbCombSrc,
        kCbComb,
        kCbPostSrc,
        kCbPost,
        kCbStreams,
        kCbSub,
        hc,
        n_tiles,
        tile_buffering,
    };
    TensorAccessorArgs(post.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(comb.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(sublayer_out.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(streams.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {kCbOut};
    TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    const std::vector<uint32_t> compute_compile_time_args = {
        kCbComb,
        kCbPost,
        kCbStreams,
        kCbSub,
        kCbOut,
        n_tiles,
    };

    const KernelHandle reader_kernel_id =
        CreateKernel(program, kReaderKernelPath, all_cores, ReaderDataMovementConfig(reader_compile_time_args));
    const KernelHandle writer_kernel_id =
        CreateKernel(program, kWriterKernelPath, all_cores, WriterDataMovementConfig(writer_compile_time_args));
    const KernelHandle compute_kernel_id = CreateKernel(
        program,
        kComputeKernelPath,
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args});

    const std::vector<CoreCoord> cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    uint32_t start_tile = 0;
    for (const auto& core : cores) {
        uint32_t tiles_this_core = 0;
        if (core_group_1.contains(core)) {
            tiles_this_core = tiles_per_core_1;
        } else if (core_group_2.contains(core)) {
            tiles_this_core = tiles_per_core_2;
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {post.buffer()->address(),
             comb.buffer()->address(),
             sublayer_out.buffer()->address(),
             streams.buffer()->address(),
             start_tile,
             tiles_this_core});
        SetRuntimeArgs(program, compute_kernel_id, core, {tiles_this_core, start_tile});
        SetRuntimeArgs(program, writer_kernel_id, core, {output.buffer()->address(), start_tile, tiles_this_core});

        start_tile += tiles_this_core;
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id, .writer_kernel_id = writer_kernel_id, .cores = cores}};
}

void MixStreamsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MixStreamsParams& /*operation_attributes*/,
    const MixStreamsInputs& tensor_args,
    MixStreamsTensorReturn& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t post_addr = tensor_args.post.buffer()->address();
    const uint32_t comb_addr = tensor_args.comb.buffer()->address();
    const uint32_t sub_addr = tensor_args.sublayer_out.buffer()->address();
    const uint32_t streams_addr = tensor_args.streams.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();

    auto& reader_args_by_core = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& reader_args = reader_args_by_core[core.x][core.y];
        reader_args[0] = post_addr;
        reader_args[1] = comb_addr;
        reader_args[2] = sub_addr;
        reader_args[3] = streams_addr;

        writer_args_by_core[core.x][core.y][0] = out_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek::mix_streams
