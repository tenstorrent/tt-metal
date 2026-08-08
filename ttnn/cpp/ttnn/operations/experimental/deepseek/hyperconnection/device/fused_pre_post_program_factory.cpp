// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_pre_post_program_factory.hpp"

#include <algorithm>
#include <bit>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace {

constexpr uint32_t kCbPreW = tt::CBIndex::c_0;
constexpr uint32_t kCbPostW = tt::CBIndex::c_1;
constexpr uint32_t kCbPreBias = tt::CBIndex::c_2;
constexpr uint32_t kCbPostBias = tt::CBIndex::c_3;
constexpr uint32_t kCbHidden = tt::CBIndex::c_4;
constexpr uint32_t kCbPostOut = tt::CBIndex::c_5;
constexpr uint32_t kCbCollapsed = tt::CBIndex::c_6;
constexpr uint32_t kCbScratch = tt::CBIndex::c_7;
constexpr uint32_t kCbPre = tt::CBIndex::c_8;
// fused_w scratch (read from DRAM, mined for pre_w / post_w / comb_w) and the comb_w_mat
// output tile (laid out as the [H,H] grid by the reader, copied to DRAM by the writer).
constexpr uint32_t kCbFusedW = tt::CBIndex::c_9;
constexpr uint32_t kCbCombW = tt::CBIndex::c_10;
// Writer scratch holding post as a column, since the op emits post as [1,T,H,1].
constexpr uint32_t kCbPostCol = tt::CBIndex::c_11;

constexpr char kReaderKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "reader_fused_pre_post.cpp";
constexpr char kComputeKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/compute/"
    "fused_pre_post.cpp";
constexpr char kWriterKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "writer_fused_pre_post.cpp";

}  // namespace

FusedPrePostProgramFactory::cached_program_t FusedPrePostProgramFactory::create(
    const FusedPrePostParams& operation_attributes,
    const FusedPrePostInputs& tensor_args,
    FusedPrePostTensorReturn& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& fused_w = tensor_args.fused_w;
    const auto& pre_bias = tensor_args.pre_bias;
    const auto& post_bias = tensor_args.post_bias;
    const auto& hidden_streams = tensor_args.hidden_streams;
    auto& post_out = tensor_return_value[0];
    auto& collapsed_out = tensor_return_value[1];
    auto& comb_w_mat_out = tensor_return_value[2];

    const uint32_t hc = operation_attributes.num_streams;

    Program program = CreateProgram();

    const DataFormat tile_data_format = datatype_to_dataformat_converter(fused_w.dtype());
    const uint32_t tile_size_bytes = tile_size(tile_data_format);

    // fused_w is [1,1,T,(2+H)*H]: token t lives in row t%32 of tile row t/32, which is
    // fused_w_row_tiles = ceil((2+H)*H/32) tiles wide. A core stages one such tile row at a
    // time and mines every token it owns out of it. pre_w / post_w / comb_w_mat are one tile
    // per token; hidden_streams contributes d_tiles per token (H <= 32, so a token's [H,D]
    // slab is a single tile row) and collapsed the same.
    const uint32_t fused_w_row_tiles = fused_w.padded_shape()[-1] / constants::TILE_WIDTH;
    const uint32_t d_tiles = hidden_streams.padded_shape()[-1] / constants::TILE_WIDTH;
    const uint32_t num_tokens = static_cast<uint32_t>(fused_w.logical_shape()[2]);

    // Tokens are fully independent here, so they are the unit of work: one core per token up
    // to the grid size. Each token costs d_tiles matmuls plus a handful of single-tile eltwise
    // ops, and nothing is shared between tokens except the two bias rows, which every core
    // re-reads for itself.
    IDevice* device = fused_w.device();
    const CoreCoord grid_size = device->compute_with_storage_grid_size();
    const uint32_t max_cores =
        std::min<uint32_t>(num_tokens, static_cast<uint32_t>(grid_size.x) * static_cast<uint32_t>(grid_size.y));
    const CoreRangeSet work_grid = num_cores_to_corerangeset(max_cores, grid_size, /*row_wise=*/true);
    const auto [num_cores, all_cores, core_group_1, core_group_2, tokens_per_core_1, tokens_per_core_2] =
        split_work_to_cores(work_grid, num_tokens, /*row_wise=*/true);

    constexpr uint32_t tile_buffering = 2;
    auto make_cb = [&](uint32_t index, uint32_t num_pages) {
        CircularBufferConfig config = CircularBufferConfig(num_pages * tile_size_bytes, {{index, tile_data_format}})
                                          .set_page_size(index, tile_size_bytes);
        CreateCircularBuffer(program, all_cores, config);
    };
    make_cb(kCbFusedW, std::max<uint32_t>(fused_w_row_tiles, 1));
    make_cb(kCbPreW, tile_buffering);
    make_cb(kCbPostW, tile_buffering);
    make_cb(kCbPreBias, 1);
    make_cb(kCbPostBias, 1);
    make_cb(kCbHidden, d_tiles);
    make_cb(kCbPostOut, tile_buffering);
    make_cb(kCbCollapsed, d_tiles);
    make_cb(kCbScratch, tile_buffering);
    make_cb(kCbPre, tile_buffering);
    make_cb(kCbCombW, tile_buffering);
    make_cb(kCbPostCol, 1);

    std::vector<uint32_t> reader_compile_time_args = {
        kCbFusedW,
        kCbPreW,
        kCbPostW,
        kCbCombW,
        kCbPreBias,
        kCbPostBias,
        kCbHidden,
        tile_buffering,
    };
    TensorAccessorArgs(fused_w.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(pre_bias.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(post_bias.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(hidden_streams.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {kCbPostOut, kCbCollapsed, kCbCombW, kCbPostCol, hc};
    TensorAccessorArgs(post_out.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(collapsed_out.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(comb_w_mat_out.buffer()).append_to(writer_compile_time_args);

    const uint32_t pre_scale_bits = std::bit_cast<uint32_t>(operation_attributes.pre_scale);
    const uint32_t post_scale_bits = std::bit_cast<uint32_t>(operation_attributes.post_scale);
    const uint32_t eps_bits = std::bit_cast<uint32_t>(operation_attributes.eps);
    const uint32_t two_bits = std::bit_cast<uint32_t>(2.0f);

    std::vector<uint32_t> compute_compile_time_args = {
        kCbPreW,
        kCbPostW,
        kCbPreBias,
        kCbPostBias,
        kCbHidden,
        kCbPostOut,
        kCbCollapsed,
        kCbScratch,
        kCbPre,
        pre_scale_bits,
        post_scale_bits,
        eps_bits,
        two_bits,
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
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    std::vector<CoreCoord> cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    uint32_t start_token = 0;
    for (const auto& core : cores) {
        uint32_t tokens_this_core = 0;
        if (core_group_1.contains(core)) {
            tokens_this_core = tokens_per_core_1;
        } else if (core_group_2.contains(core)) {
            tokens_this_core = tokens_per_core_2;
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {fused_w.buffer()->address(),
             pre_bias.buffer()->address(),
             post_bias.buffer()->address(),
             hidden_streams.buffer()->address(),
             hc,
             fused_w_row_tiles,
             d_tiles,
             start_token,
             tokens_this_core});
        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {post_out.buffer()->address(),
             collapsed_out.buffer()->address(),
             comb_w_mat_out.buffer()->address(),
             d_tiles,
             start_token,
             tokens_this_core});
        SetRuntimeArgs(program, compute_kernel_id, core, {d_tiles, tokens_this_core});

        start_token += tokens_this_core;
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .cores = std::move(cores)}};
}

void FusedPrePostProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FusedPrePostParams& /*operation_attributes*/,
    const FusedPrePostInputs& tensor_args,
    FusedPrePostTensorReturn& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t fused_w_addr = tensor_args.fused_w.buffer()->address();
    const uint32_t pre_bias_addr = tensor_args.pre_bias.buffer()->address();
    const uint32_t post_bias_addr = tensor_args.post_bias.buffer()->address();
    const uint32_t hidden_addr = tensor_args.hidden_streams.buffer()->address();
    const uint32_t post_addr = tensor_return_value[0].buffer()->address();
    const uint32_t collapsed_addr = tensor_return_value[1].buffer()->address();
    const uint32_t comb_w_addr = tensor_return_value[2].buffer()->address();

    auto& reader_args_by_core = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& reader_args = reader_args_by_core[core.x][core.y];
        reader_args[0] = fused_w_addr;
        reader_args[1] = pre_bias_addr;
        reader_args[2] = post_bias_addr;
        reader_args[3] = hidden_addr;

        auto& writer_args = writer_args_by_core[core.x][core.y];
        writer_args[0] = post_addr;
        writer_args[1] = collapsed_addr;
        writer_args[2] = comb_w_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
