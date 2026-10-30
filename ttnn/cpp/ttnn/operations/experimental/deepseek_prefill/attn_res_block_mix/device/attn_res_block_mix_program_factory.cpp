// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/attn_res_block_mix_program_factory.hpp"

#include <algorithm>
#include <bit>

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

constexpr auto kKernelDir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/kernels/";

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

}  // namespace

tt::tt_metal::ProgramDescriptor AttnResBlockMixProgramFactory::create_descriptor(
    const AttnResBlockMixParams& operation_attributes,
    const AttnResBlockMixInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& stats = tensor_args.stats;
    const auto& first = tensor_args.sealed.front();
    auto* device = stats.device();
    auto& [partials, shift, mass] = tensor_return_value;

    const auto value_format = datatype_to_dataformat_converter(first.dtype());
    const auto value_tile_size = tt::tile_size(value_format);
    const auto stats_format = datatype_to_dataformat_converter(stats.dtype());
    const auto stats_tile_size = tt::tile_size(stats_format);

    const auto& shape = first.padded_shape();
    const uint32_t Ht = shape[2] / TILE_HEIGHT;
    const uint32_t Wt = shape[3] / TILE_WIDTH;
    const uint32_t num_candidates = stats.padded_shape()[1];
    const uint32_t num_partials = stats.padded_shape()[3] / TILE_WIDTH;
    const uint32_t num_sites = operation_attributes.num_sites;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // A group of sites is one dest session, one accumulator per site, so the group is as
    // large as dest holds. Every group MACs the same candidate tiles, which stay in L1 until
    // the last group is done with them: the sealed set is read once whatever the group count.
    const uint32_t sites_per_group =
        std::min(num_sites, get_dest_reg_count(operation_attributes.compute_kernel_config));
    const uint32_t num_groups = (num_sites + sites_per_group - 1) / sites_per_group;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // A work item is a run of tiles along one token row-tile. Splitting rows alone would put
    // a few dozen cores behind every write of the R partials; splitting each row into chunks
    // spreads them over the grid. Each core of a row scores that row itself, a few tiles of
    // arithmetic, rather than wait on another core for it.
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const uint32_t chunks_per_row = std::clamp(grid_cores / Ht, 1u, Wt);
    const uint32_t num_items = Ht * chunks_per_row;
    const uint32_t num_cores = std::min(num_items, grid_cores);
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    // A run's candidate tiles stay resident for every site group, and a group's weights are
    // double-buffered so the next group is built while this one is mixed.
    const uint32_t max_run = (Wt + chunks_per_row - 1) / chunks_per_row;
    const uint32_t value_tiles = max_run * num_candidates;
    const uint32_t weight_tiles = 2 * sites_per_group * num_candidates;
    const uint32_t l1_per_core = num_candidates * num_partials * stats_tile_size  // gathered statistics
                                 + 3 * num_candidates * stats_tile_size           // summed, rsqrt, scores
                                 + 5 * stats_tile_size                            // shift, shift and mass out
                                 + num_candidates * value_tile_size               // packed exponentials
                                 + value_tiles * value_tile_size                  // the run's candidate tiles
                                 + weight_tiles * value_tile_size                 // per-site weights
                                 + 2 * sites_per_group * value_tile_size          // partials out
                                 + num_sites * stats_tile_size;                   // scalar staging
    TT_FATAL(
        l1_per_core <= device->l1_size_per_core(),
        "AttnResBlockMix needs {} B of L1 per core for {} sites over {} candidates; the core has {} B",
        l1_per_core,
        num_sites,
        num_candidates,
        device->l1_size_per_core());

    ProgramDescriptor desc;
    const auto cb = [&](CBIndex index, uint32_t tiles, uint32_t tile_size, DataFormat format) {
        desc.cbs.push_back(make_cb(index, all_cores, tiles, tile_size, format));
    };
    // Scoring, in the statistics tile's own layout: a row is a token, a column a site.
    cb(CBIndex::c_0, num_candidates * num_partials, stats_tile_size, stats_format);  // gathered statistics
    cb(CBIndex::c_1, num_candidates, stats_tile_size, stats_format);                 // rsqrt
    cb(CBIndex::c_2, num_candidates, stats_tile_size, stats_format);                 // scores
    cb(CBIndex::c_3, 1, stats_tile_size, stats_format);                              // shift, for compute
    cb(CBIndex::c_4, num_candidates, stats_tile_size, stats_format);                 // summed statistics
    cb(CBIndex::c_5, num_candidates, value_tile_size, value_format);                 // packed exponentials
    // The mixture: a site's weight is column 0 of its own tile, the column a broadcast MAC
    // reads; the reader builds them from the packed exponentials.
    cb(CBIndex::c_6, value_tiles, value_tile_size, value_format);           // the run's candidate tiles
    cb(CBIndex::c_7, weight_tiles, value_tile_size, value_format);          // a group's weights, twice
    cb(CBIndex::c_16, 2, stats_tile_size, stats_format);                    // shift, for the writer
    cb(CBIndex::c_17, 2, stats_tile_size, stats_format);                    // mass
    cb(CBIndex::c_18, 2 * sites_per_group, value_tile_size, value_format);  // partials
    cb(CBIndex::c_24, num_sites, stats_tile_size, stats_format);            // scalar staging, never pushed

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels
    ////////////////////////////////////////////////////////////////////////////
    // A core finds its items from its own position, so a launch ships no per-core args.
    const std::vector<uint32_t> split_ct_args = {grid.x, num_cores, num_items, chunks_per_row, Wt};

    std::vector<uint32_t> reader_ct_args = {num_candidates, num_partials, Ht, num_sites, sites_per_group, num_groups};
    reader_ct_args.insert(reader_ct_args.end(), split_ct_args.begin(), split_ct_args.end());
    TensorAccessorArgs(*stats.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*first.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {num_sites, Ht, sites_per_group, num_groups};
    writer_ct_args.insert(writer_ct_args.end(), split_ct_args.begin(), split_ct_args.end());
    TensorAccessorArgs(*partials.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*shift.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*mass.buffer()).append_to(writer_ct_args);

    std::vector<uint32_t> compute_ct_args = {
        num_candidates,
        num_partials,
        num_sites,
        std::bit_cast<uint32_t>(operation_attributes.inv_hidden_size),
        std::bit_cast<uint32_t>(operation_attributes.eps),
        sites_per_group,
        num_groups};
    compute_ct_args.insert(compute_ct_args.end(), split_ct_args.begin(), split_ct_args.end());

    // Each candidate is a (snapshot address, first page) pair: the snapshot that holds it and
    // the offset of its plane there. The snapshots share one accessor configuration, which
    // the device operation checks, so the address is all that differs between them.
    KernelDescriptor::RTArgList reader_common;
    reader_common.push_back(stats.buffer());
    for (const auto& snapshot : tensor_args.sealed) {
        for (uint32_t plane = 0; plane < snapshot.padded_shape()[1]; ++plane) {
            reader_common.push_back(snapshot.buffer());
            reader_common.push_back(plane * Ht * Wt);
        }
    }

    KernelDescriptor reader;
    reader.kernel_source = std::string(kKernelDir) + "reader_attn_res_block_mix.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct_args);
    reader.config = ReaderConfigDescriptor{};
    reader.emplace_common_runtime_args(reader_common);

    KernelDescriptor writer;
    writer.kernel_source = std::string(kKernelDir) + "writer_attn_res_block_mix.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct_args);
    writer.config = WriterConfigDescriptor{};
    writer.emplace_common_runtime_args({partials.buffer(), shift.buffer(), mass.buffer()});

    // The scores and the shift are only ever copied into dest, never fed to the FPU, so they
    // unpack straight to it rather than being cut to tf32 on the way through srcA.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[CBIndex::c_2] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[CBIndex::c_3] = UnpackToDestMode::UnpackToDestFp32;

    KernelDescriptor compute;
    compute.kernel_source = std::string(kKernelDir) + "attn_res_block_mix.cpp";
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
