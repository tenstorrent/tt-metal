// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_untilize.hpp"

#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

#include "kernels/dataflow/dispatch_fabric2d_untilize_args.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

namespace {

// Tiles the untilizer packs per call: the largest divisor of the tiles per tile row that is at most eight,
// the packer's limit. pack_untilize puts block i at column offset i * block_ct_dim, so it must be a
// divisor; a smaller last block would overlap the one before it.
uint32_t untilize_block_ct_dim(uint32_t tiles_per_row) {
    uint32_t block = 8;
    while (block > 1 && tiles_per_row % block != 0) {
        block--;
    }
    return block;
}

constexpr const char* kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_fabric2d/device/kernels/";

}  // namespace

std::optional<UntilizePlan> plan_untilize(
    const ttnn::Tensor& input,
    const ttnn::Tensor& out_payload,
    uint32_t seq_len_per_chip,
    uint32_t token_bytes,
    uint32_t sem_addr,
    tt::tt_metal::Buffer* staging) {
    if (input.layout() != tt::tt_metal::Layout::TILE) {
        return std::nullopt;
    }
    UntilizePlan plan;
    plan.tiles_per_row = static_cast<uint32_t>(input.logical_shape()[-1]) / tt::constants::TILE_WIDTH;
    plan.block_ct_dim = untilize_block_ct_dim(plan.tiles_per_row);
    // A partial last tile row is packed whole, including the tile's padding rows. Staging is sized to
    // match, so those rows land in pages past the sequence that nothing reads.
    plan.num_tile_rows = (seq_len_per_chip + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
    plan.tile_bytes = static_cast<uint32_t>(input.buffer()->aligned_page_size());
    plan.token_bytes = token_bytes;
    plan.sem_addr = sem_addr;
    plan.tile_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    plan.row_format = tt::tt_metal::datatype_to_dataformat_converter(out_payload.dtype());
    plan.input = input.buffer();
    plan.staging = staging;

    // The packer writes rows at a stride of tiles_per_row * TILE_WIDTH elements in the payload's format,
    // while the writer and the staging page count a row as token_bytes. They agree only while a row needs
    // no alignment padding; otherwise every token after the first in a tile row is shifted, and nothing
    // later detects it.
    const uint32_t packed_row_bytes =
        plan.tiles_per_row * tt::constants::TILE_WIDTH * static_cast<uint32_t>(out_payload.element_size());
    TT_FATAL(
        packed_row_bytes == token_bytes,
        "dispatch_fabric2d: the untilizer packs a row as {} B but a token page is {} B",
        packed_row_bytes,
        token_bytes);
    return plan;
}

UntilizerPoolFallback add_untilizer_pool(
    tt::tt_metal::ProgramDescriptor& desc,
    const StreamPlacements& streams,
    const CoreRangeSet& allowed_cores,
    const UntilizePlan& plan) {
    bool spilled = false;
    const auto pool = decide_untilizer_pool(allowed_cores, streams, plan.num_tile_rows, UNTILIZERS_PER_LINK, &spilled);
    const UntilizerPoolFallback fallback =
        spilled ? UntilizerPoolFallback::kRowTooNarrow : UntilizerPoolFallback::kNone;
    const uint32_t pool_size = static_cast<uint32_t>(pool.size());
    const CoreRangeSet pool_cores(ttsl::Span<const tt::tt_metal::CoreCoord>(pool.data(), pool_size));

    // One source for the two CB indices: the descriptors below and the kernels' compile-time args must name
    // the same buffers.
    constexpr auto kTileCbIndex = tt::CBIndex::c_0;
    constexpr auto kRowCbIndex = tt::CBIndex::c_11;

    // Tiles, reader -> compute. Two blocks of block_ct_dim tiles: a whole number of blocks, so a block
    // never wraps around the end of the CB, and two so the reader can work one block ahead of the packer.
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = 2u * plan.block_ct_dim * plan.tile_bytes,
        .core_ranges = pool_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(kTileCbIndex),
            .data_format = plan.tile_format,
            .page_size = plan.tile_bytes,
        }}},
    });
    // Untilized rows, compute -> writer. A whole number of tile rows, so a tile row's rows are one
    // contiguous run; pack_untilize writes each column block at an offset into that run. The format is
    // the payload's, because the packer converts as it writes.
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = 2u * tt::constants::TILE_HEIGHT * plan.token_bytes,
        .core_ranges = pool_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(kRowCbIndex),
            .data_format = plan.row_format,
            .page_size = plan.token_bytes,
        }}},
    });

    // Virtual, because this is what a NoC write off this core addresses.
    std::vector<uint32_t> stream_coords;
    for (const auto& [stream, placement] : streams) {
        stream_coords.push_back(static_cast<uint32_t>(placement.worker_virtual.x));
        stream_coords.push_back(static_cast<uint32_t>(placement.worker_virtual.y));
    }
    const dspf2d::UntilizeCtArgs args(
        plan,
        static_cast<uint32_t>(kTileCbIndex),
        static_cast<uint32_t>(kRowCbIndex),
        pool_size,
        static_cast<uint32_t>(streams.size()));
    const std::vector<uint32_t> ct = args.to_ct_word_arr(stream_coords);

    tt::tt_metal::KernelDescriptor rdr;
    rdr.kernel_source = std::string(kKernelDir) + "dataflow/untilize_reader_dispatch_fabric2d.cpp";
    rdr.core_ranges = pool_cores;
    rdr.compile_time_args = ct;
    tt::tt_metal::TensorAccessorArgs(plan.input).append_to(rdr.compile_time_args);
    rdr.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::NOC_0,
    };

    tt::tt_metal::KernelDescriptor cmp;
    cmp.kernel_source = std::string(kKernelDir) + "compute/untilize_dispatch_fabric2d.cpp";
    cmp.core_ranges = pool_cores;
    cmp.compile_time_args = ct;
    cmp.config = tt::tt_metal::ComputeConfigDescriptor{.math_fidelity = tt::tt_metal::MathFidelity::HiFi4};

    tt::tt_metal::KernelDescriptor wtr;
    wtr.kernel_source = std::string(kKernelDir) + "dataflow/untilize_writer_dispatch_fabric2d.cpp";
    wtr.core_ranges = pool_cores;
    wtr.compile_time_args = ct;
    tt::tt_metal::TensorAccessorArgs(plan.staging).append_to(wtr.compile_time_args);
    wtr.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::NOC_1,
    };

    // The pool index is the only per-core value. It is a runtime argument because a compile-time one
    // would build a separate binary per core.
    for (uint32_t i = 0; i < pool_size; i++) {
        tt::tt_metal::KernelDescriptor::RTArgList rdr_rt;
        rdr_rt.push_back(i);
        rdr_rt.push_back(plan.input);
        rdr.emplace_runtime_args(pool[i], rdr_rt);

        tt::tt_metal::KernelDescriptor::RTArgList cmp_rt;
        cmp_rt.push_back(i);
        cmp.emplace_runtime_args(pool[i], cmp_rt);

        tt::tt_metal::KernelDescriptor::RTArgList wtr_rt;
        wtr_rt.push_back(i);
        wtr_rt.push_back(plan.staging);
        wtr.emplace_runtime_args(pool[i], wtr_rt);
    }
    desc.kernels.push_back(std::move(rdr));
    desc.kernels.push_back(std::move(cmp));
    desc.kernels.push_back(std::move(wtr));
    return fallback;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
