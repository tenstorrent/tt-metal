// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_untilize.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

#include "kernels/dataflow/dispatch_fabric2d_untilize_args.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

namespace {

// Tile columns the untilizer packs per call: the largest divisor of a stripe's tile count that is at
// most eight. pack_untilize computes a block's L1 column offset as block_index * block_ct_dim, so a
// remainder block would land on top of the previous one -- a divisor is the requirement, not a
// preference, and eight is where the packer's own tile budget ends.
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
    plan.num_stripes = seq_len_per_chip / tt::constants::TILE_HEIGHT;
    plan.tile_bytes = static_cast<uint32_t>(input.buffer()->aligned_page_size());
    plan.token_bytes = token_bytes;
    plan.sem_addr = sem_addr;
    plan.tile_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    plan.row_format = tt::tt_metal::datatype_to_dataformat_converter(out_payload.dtype());
    plan.input = input.buffer();
    plan.staging = staging;

    // The packer writes its rows at a stride of tiles_per_row * TILE_WIDTH datums in the PAYLOAD's
    // format, while the writer beside it and the staging page both count a row as token_bytes. Those
    // agree only while the row needs no alignment padding; if they ever diverge, every token after the
    // first of a stripe shears by the difference, which is the one failure this path has no way to
    // notice.
    const uint32_t packed_row_bytes =
        plan.tiles_per_row * tt::constants::TILE_WIDTH * static_cast<uint32_t>(out_payload.element_size());
    TT_FATAL(
        packed_row_bytes == token_bytes,
        "dispatch_fabric2d: the untilizer packs a row as {} B but a token page is {} B",
        packed_row_bytes,
        token_bytes);
    return plan;
}

void add_untilizer_pool(
    tt::tt_metal::ProgramDescriptor& desc,
    const StreamPlacements& streams,
    const CoreRangeSet& universe,
    const UntilizePlan& plan) {
    const auto spare = spare_cores(universe, streams);
    TT_FATAL(!spare.empty(), "dispatch_fabric2d: a TILE input needs at least one core beside the streams");
    // No point in more cores than stripes; a core with no stripes would still pay its program build.
    const uint32_t pool_size = std::min<uint32_t>(static_cast<uint32_t>(spare.size()), plan.num_stripes);
    const CoreRangeSet pool_cores(ttsl::Span<const tt::tt_metal::CoreCoord>(spare.data(), pool_size));

    // Tiled stripe, reader -> compute. A whole number of block_ct_dim blocks deep, so a block never
    // straddles the ring wrap, and two blocks deep so the reader runs ahead of the packer.
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = 2u * plan.block_ct_dim * plan.tile_bytes,
        .core_ranges = pool_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = plan.tile_format,
            .page_size = plan.tile_bytes,
        }}},
    });
    // Untilized rows, compute -> writer. A whole number of stripes, so a stripe's rows are one
    // contiguous run -- pack_untilize writes each column block at an offset into that run. The index
    // is c_11 to match the sibling `dispatch` op's untilize output, so a profile of the two reads the
    // same. Its format is the payload's rather than the input's: the packer converts as it writes here,
    // which is where that op puts its BFLOAT16 -> FP8 conversion too.
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = 2u * tt::constants::TILE_HEIGHT * plan.token_bytes,
        .core_ranges = pool_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_11),
            .data_format = plan.row_format,
            .page_size = plan.token_bytes,
        }}},
    });

    std::vector<uint32_t> ct(dspf2d::UntilizeCtArgs::kCount, 0u);
    ct[dspf2d::UntilizeCtArgs::kTileCb] = static_cast<uint32_t>(tt::CBIndex::c_0);
    ct[dspf2d::UntilizeCtArgs::kRowCb] = static_cast<uint32_t>(tt::CBIndex::c_11);
    ct[dspf2d::UntilizeCtArgs::kNumStripes] = plan.num_stripes;
    ct[dspf2d::UntilizeCtArgs::kPoolSize] = pool_size;
    ct[dspf2d::UntilizeCtArgs::kTilesPerRow] = plan.tiles_per_row;
    ct[dspf2d::UntilizeCtArgs::kBlockCtDim] = plan.block_ct_dim;
    ct[dspf2d::UntilizeCtArgs::kTileBytes] = plan.tile_bytes;
    ct[dspf2d::UntilizeCtArgs::kTokenBytes] = plan.token_bytes;
    ct[dspf2d::UntilizeCtArgs::kRowsPerStripe] = tt::constants::TILE_HEIGHT;
    ct[dspf2d::UntilizeCtArgs::kStreamCount] = static_cast<uint32_t>(streams.size());
    ct[dspf2d::UntilizeCtArgs::kUntilizeSemAddr] = plan.sem_addr;
    ct[dspf2d::UntilizeCtArgs::kStreamCoordsBase] = dspf2d::UntilizeCtArgs::kCount;
    // Virtual, because this is what a NoC write off this core addresses.
    for (const auto& [stream, placement] : streams) {
        ct.push_back(static_cast<uint32_t>(placement.worker_virtual.x));
        ct.push_back(static_cast<uint32_t>(placement.worker_virtual.y));
    }
    // The kernels index the coordinates from this base and chain their accessor arguments past them.
    // A base that does not match the block actually appended sends every stripe's arrival to a core
    // that is not there, which the stream readers then wait for forever.
    TT_FATAL(
        ct.size() == dspf2d::UntilizeCtArgs::kCount + 2u * streams.size(),
        "dispatch_fabric2d: untilizer compile-time args are {} words but the kernels index {}",
        ct.size(),
        dspf2d::UntilizeCtArgs::kCount + 2u * streams.size());

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
    cmp.config = tt::tt_metal::ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4};

    tt::tt_metal::KernelDescriptor wtr;
    wtr.kernel_source = std::string(kKernelDir) + "dataflow/untilize_writer_dispatch_fabric2d.cpp";
    wtr.core_ranges = pool_cores;
    wtr.compile_time_args = ct;
    tt::tt_metal::TensorAccessorArgs(plan.staging).append_to(wtr.compile_time_args);
    wtr.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::NOC_1,
    };

    // The only thing that differs between the pool's cores, so the only thing held per core: as a
    // compile-time argument it would build a separate binary for each of them.
    for (uint32_t i = 0; i < pool_size; i++) {
        tt::tt_metal::KernelDescriptor::RTArgList rdr_rt;
        rdr_rt.push_back(i);
        rdr_rt.push_back(plan.input);
        rdr.emplace_runtime_args(spare[i], rdr_rt);

        tt::tt_metal::KernelDescriptor::RTArgList cmp_rt;
        cmp_rt.push_back(i);
        cmp.emplace_runtime_args(spare[i], cmp_rt);

        tt::tt_metal::KernelDescriptor::RTArgList wtr_rt;
        wtr_rt.push_back(i);
        wtr_rt.push_back(plan.staging);
        wtr.emplace_runtime_args(spare[i], wtr_rt);
    }
    desc.kernels.push_back(std::move(rdr));
    desc.kernels.push_back(std::move(cmp));
    desc.kernels.push_back(std::move(wtr));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
