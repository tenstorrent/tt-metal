// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/attn_res_block_stats_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr auto kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/kernels/";

CBDescriptor make_cb(
    CBIndex index, const CoreRangeSet& cores, uint32_t num_tiles, uint32_t tile_size, DataFormat format) {
    return CBDescriptor{
        .total_size = num_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = format,
            .page_size = tile_size,
        }}},
    };
}

// The most runs a row is cut into: a row's leader holds a partial per run and candidate.
constexpr uint32_t kMaxChunksPerRow = 8;

}  // namespace

tt::tt_metal::ProgramDescriptor AttnResBlockStatsProgramFactory::create_descriptor(
    const AttnResBlockStatsParams& operation_attributes,
    const AttnResBlockStatsInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* device = tensor_args.queries.device();
    const auto& first = tensor_args.sealed.front();

    const auto in_format = datatype_to_dataformat_converter(first.dtype());
    const auto in_tile_size = tt::tile_size(in_format);
    const auto out_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto out_tile_size = tt::tile_size(out_format);

    const auto& shape = first.padded_shape();
    const uint32_t Ht = shape[2] / TILE_HEIGHT;
    const uint32_t Wt = shape[3] / TILE_WIDTH;
    const uint32_t num_candidates = tensor_return_value.padded_shape()[1];

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // A unit is a run of one token row across every candidate (see
    // `kernels/attn_res_block_stats_items.hpp`). The op is bound by how fast the grid pulls
    // the sealed set in, so rows are cut into as many runs as the grid seats and every core
    // reads the same share; whole (candidate, row) items left a grid that does not divide
    // them reading a second wave on a few cores. Where rows outnumber the cores, a core
    // takes several whole rows.
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const uint32_t chunks_per_row = std::clamp(grid_cores / Ht, 1u, std::min(Wt, kMaxChunksPerRow));
    const uint32_t num_units = Ht * chunks_per_row;
    const uint32_t units_per_core = (num_units + grid_cores - 1) / grid_cores;
    const uint32_t num_cores = (num_units + units_per_core - 1) / units_per_core;
    const uint32_t max_run = (Wt + chunks_per_row - 1) / chunks_per_row;
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const auto cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    TT_FATAL(chunks_per_row == 1 || units_per_core == 1, "a cut row must take one core per run");

    ProgramDescriptor desc;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    // A candidate's run is one CB unit, double-buffered so the next candidate's run arrives
    // while this one is reduced. The squares hold a run too: the sum of squares is a second
    // matmul over them, into the same accumulator as the dots.
    desc.cbs.push_back(make_cb(CBIndex::c_0, all_cores, 2 * max_run, in_tile_size, in_format));  // v
    desc.cbs.push_back(make_cb(CBIndex::c_1, all_cores, max_run, in_tile_size, in_format));      // queries, resident
    desc.cbs.push_back(make_cb(CBIndex::c_2, all_cores, 1, in_tile_size, in_format));            // ones in column 0
    desc.cbs.push_back(make_cb(CBIndex::c_3, all_cores, max_run, in_tile_size, in_format));      // v squared
    desc.cbs.push_back(make_cb(CBIndex::c_16, all_cores, 2, out_tile_size, out_format));         // statistics
    if (chunks_per_row > 1) {
        // A row leader's own partials, and the row's other runs', written in by them. On
        // every core, so a sender finds the leader's at its own copy's address.
        desc.cbs.push_back(make_cb(CBIndex::c_4, all_cores, num_candidates, out_tile_size, out_format));
        desc.cbs.push_back(
            make_cb(CBIndex::c_5, all_cores, (chunks_per_row - 1) * num_candidates, out_tile_size, out_format));
    }
    // A row leader's count of its other runs.
    constexpr uint32_t kReceivedSem = 0;
    desc.semaphores.push_back(SemaphoreDescriptor{.id = kReceivedSem, .core_ranges = all_cores, .initial_value = 0});

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels
    ////////////////////////////////////////////////////////////////////////////
    // A core finds its units from its own position, so a launch ships no per-core args.
    const std::vector<uint32_t> split_ct_args = {grid.x, units_per_core, num_units, chunks_per_row, Wt};

    std::vector<uint32_t> reader_ct_args = {Ht, num_candidates};
    reader_ct_args.insert(reader_ct_args.end(), split_ct_args.begin(), split_ct_args.end());
    TensorAccessorArgs(*first.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*tensor_args.queries.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {Ht, num_candidates, kReceivedSem};
    writer_ct_args.insert(writer_ct_args.end(), split_ct_args.begin(), split_ct_args.end());
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_ct_args);

    std::vector<uint32_t> compute_ct_args = {num_candidates};
    compute_ct_args.insert(compute_ct_args.end(), split_ct_args.begin(), split_ct_args.end());

    // Each candidate is a (snapshot address, first page) pair: the snapshot that holds it and
    // the offset of its plane there. The snapshots share one accessor configuration, which
    // the device operation checks, so the address is all that differs between them.
    KernelDescriptor::RTArgList reader_common;
    reader_common.push_back(tensor_args.queries.buffer());
    for (const auto& snapshot : tensor_args.sealed) {
        for (uint32_t plane = 0; plane < snapshot.padded_shape()[1]; ++plane) {
            reader_common.push_back(snapshot.buffer());
            reader_common.push_back(plane * Ht * Wt);
        }
    }

    // Where rows are cut, each row's leader — its chunk 0 core — that the row's runs send to.
    KernelDescriptor::RTArgList writer_common;
    writer_common.push_back(tensor_return_value.buffer());
    if (chunks_per_row > 1) {
        for (uint32_t g = 0; g < Ht; ++g) {
            const auto leader = device->worker_core_from_logical_core(cores.at(g * chunks_per_row));
            writer_common.push_back((leader.x << 16) | leader.y);
        }
    }

    KernelDescriptor reader;
    reader.kernel_source = std::string(kKernelDir) + "reader_attn_res_block_stats.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct_args);
    reader.config = ReaderConfigDescriptor{};
    reader.emplace_common_runtime_args(reader_common);

    KernelDescriptor writer;
    writer.kernel_source = std::string(kKernelDir) + "writer_attn_res_block_stats.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct_args);
    writer.config = WriterConfigDescriptor{};
    writer.emplace_common_runtime_args(writer_common);

    // A leader's partials are only ever copied into dest, so they unpack straight to it
    // rather than being cut to tf32 on the way through srcA.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[CBIndex::c_4] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[CBIndex::c_5] = UnpackToDestMode::UnpackToDestFp32;

    KernelDescriptor compute;
    compute.kernel_source = std::string(kKernelDir) + "attn_res_block_stats.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = std::move(compute_ct_args);
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        .math_approx_mode = math_approx_mode,
    };

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));

    return desc;
}

}  // namespace ttnn::experimental::prim
