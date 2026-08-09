// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sinkhorn_program_factory.hpp"

#include <algorithm>
#include <bit>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace {

constexpr uint32_t kSinkCbCombW = tt::CBIndex::c_0;
constexpr uint32_t kSinkCbCombBias = tt::CBIndex::c_1;
constexpr uint32_t kSinkCbScaler = tt::CBIndex::c_2;
constexpr uint32_t kSinkCbMask = tt::CBIndex::c_3;
constexpr uint32_t kSinkCbComb = tt::CBIndex::c_4;
constexpr uint32_t kSinkCbReduce = tt::CBIndex::c_5;
constexpr uint32_t kSinkCbEpsMask = tt::CBIndex::c_6;
constexpr uint32_t kSinkCbOut = tt::CBIndex::c_7;

constexpr char kSinkhornReaderKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "reader_sinkhorn.cpp";
constexpr char kSinkhornComputeKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/compute/"
    "sinkhorn.cpp";
constexpr char kSinkhornWriterKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "writer_sinkhorn.cpp";

}  // namespace

SinkhornProgramFactory::cached_program_t SinkhornProgramFactory::create(
    const SinkhornParams& operation_attributes,
    const SinkhornInputs& tensor_args,
    SinkhornTensorReturn& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& comb_w = tensor_args.comb_w;
    const auto& comb_bias = tensor_args.comb_bias;
    auto& comb_out = tensor_return_value;

    Program program = CreateProgram();

    const DataFormat tile_data_format = datatype_to_dataformat_converter(comb_w.dtype());
    const uint32_t tile_size_bytes = tile_size(tile_data_format);

    // comb_w is [1,T,H,H] with H <= 32: one independent [H,H] tile per token, so tiles are the
    // unit of work. Every core regenerates the shared scaler / mask tiles and re-reads the bias.
    const uint32_t num_tiles = static_cast<uint32_t>(comb_w.logical_shape()[1]);

    IDevice* device = comb_w.device();
    const CoreCoord grid_size = device->compute_with_storage_grid_size();
    const uint32_t max_cores =
        std::min<uint32_t>(num_tiles, static_cast<uint32_t>(grid_size.x) * static_cast<uint32_t>(grid_size.y));
    const CoreRangeSet work_grid = num_cores_to_corerangeset(max_cores, grid_size, /*row_wise=*/true);
    const auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2] =
        split_work_to_cores(work_grid, num_tiles, /*row_wise=*/true);

    constexpr uint32_t tile_buffering = 2;
    auto make_cb = [&](uint32_t index, uint32_t num_pages) {
        CircularBufferConfig config = CircularBufferConfig(num_pages * tile_size_bytes, {{index, tile_data_format}})
                                          .set_page_size(index, tile_size_bytes);
        CreateCircularBuffer(program, all_cores, config);
    };
    make_cb(kSinkCbCombW, tile_buffering);
    make_cb(kSinkCbCombBias, 1);
    make_cb(kSinkCbScaler, 1);
    make_cb(kSinkCbMask, 1);
    make_cb(kSinkCbComb, tile_buffering);
    make_cb(kSinkCbReduce, tile_buffering);
    make_cb(kSinkCbEpsMask, 1);
    make_cb(kSinkCbOut, tile_buffering);

    // Scaler tile holds 1.0f (bf16 hi-half) for plain sum/max reductions.
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(1.0f);
    const uint32_t comb_scale_bits = std::bit_cast<uint32_t>(operation_attributes.comb_scale);
    const uint32_t eps_bits = std::bit_cast<uint32_t>(operation_attributes.eps);

    std::vector<uint32_t> reader_compile_time_args = {
        kSinkCbCombW,
        kSinkCbCombBias,
        kSinkCbMask,
        kSinkCbScaler,
        scaler_bits,
        operation_attributes.num_streams,
        kSinkCbEpsMask,
        eps_bits,
    };
    TensorAccessorArgs(comb_w.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(comb_bias.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {kSinkCbOut};
    TensorAccessorArgs(comb_out.buffer()).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        kSinkCbCombW,
        kSinkCbCombBias,
        kSinkCbScaler,
        kSinkCbMask,
        kSinkCbComb,
        kSinkCbReduce,
        kSinkCbEpsMask,
        kSinkCbOut,
        operation_attributes.num_streams,
        operation_attributes.sinkhorn_iters,
        comb_scale_bits,
        eps_bits,
    };

    const KernelHandle reader_kernel_id =
        CreateKernel(program, kSinkhornReaderKernelPath, all_cores, ReaderDataMovementConfig(reader_compile_time_args));

    const KernelHandle writer_kernel_id =
        CreateKernel(program, kSinkhornWriterKernelPath, all_cores, WriterDataMovementConfig(writer_compile_time_args));

    const KernelHandle compute_kernel_id = CreateKernel(
        program,
        kSinkhornComputeKernelPath,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    std::vector<CoreCoord> cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
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
            {comb_w.buffer()->address(), comb_bias.buffer()->address(), start_tile, tiles_this_core});
        SetRuntimeArgs(program, writer_kernel_id, core, {comb_out.buffer()->address(), start_tile, tiles_this_core});
        SetRuntimeArgs(program, compute_kernel_id, core, {tiles_this_core});

        start_tile += tiles_this_core;
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .cores = std::move(cores)}};
}

void SinkhornProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const SinkhornParams& /*operation_attributes*/,
    const SinkhornInputs& tensor_args,
    SinkhornTensorReturn& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t comb_w_addr = tensor_args.comb_w.buffer()->address();
    const uint32_t comb_bias_addr = tensor_args.comb_bias.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();

    auto& reader_args_by_core = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& reader_args = reader_args_by_core[core.x][core.y];
        reader_args[0] = comb_w_addr;
        reader_args[1] = comb_bias_addr;

        writer_args_by_core[core.x][core.y][0] = out_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
