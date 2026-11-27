// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/attn_res_block_mix_program_factory.hpp"

#include <algorithm>
#include <bit>

#include <tt-metalium/memory/buffer/buffer.hpp>
#include <tt-metalium/mesh/allocator.hpp>
#include <tt-metalium/workload/legacy/circular_buffer_constants.h>
#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/workload/legacy/program_descriptors.hpp>
#include <tt-metalium/workload/legacy/tensor_accessor_args.hpp>
#include <tt-metalium/utils/work_split.hpp>

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

// Circular buffers. The kernels take every id as a named compile-time arg.
constexpr auto cb_stats = CBIndex::c_0;   // gathered statistics
constexpr auto cb_rr = CBIndex::c_1;      // rsqrt
constexpr auto cb_scores = CBIndex::c_2;  // scores
constexpr auto cb_shift = CBIndex::c_3;   // shift, for compute
constexpr auto cb_summed = CBIndex::c_4;  // summed statistics
constexpr auto cb_exp = CBIndex::c_5;     // packed exponentials
constexpr auto cb_values = CBIndex::c_6;  // the run's candidate tiles
constexpr auto cb_weights = CBIndex::c_7;
constexpr auto cb_mass = CBIndex::c_8;           // mass, for compute
constexpr auto cb_closed_mask = CBIndex::c_9;    // closed-column mask
constexpr auto cb_factor = CBIndex::c_10;        // per-site factor
constexpr auto cb_shift_out = CBIndex::c_16;     // shift, for the writer
constexpr auto cb_mass_out = CBIndex::c_17;      // mass, for the writer
constexpr auto cb_partials = CBIndex::c_18;      // even site groups
constexpr auto cb_partials_odd = CBIndex::c_19;  // odd site groups
constexpr auto cb_staging = CBIndex::c_24;       // scalar staging, never pushed

}  // namespace

tt::tt_metal::ProgramDescriptor AttnResBlockMixProgramFactory::create_descriptor(
    const AttnResBlockMixParams& operation_attributes,
    const AttnResBlockMixInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& stats = tensor_args.stats;
    const auto& first = tensor_args.sealed.front();
    auto* device = stats.device();
    auto& partials = tensor_return_value.at(0);
    auto& shift = tensor_return_value.at(1);
    auto& mass = tensor_return_value.at(2);

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
    const uint32_t num_closed_sites = operation_attributes.num_closed_sites;
    const uint32_t num_open_sites = num_sites - num_closed_sites;
    // Each partial is over one tensor-parallel shard of `d`, so together they cover the
    // hidden size the mean square is taken over.
    const float inv_hidden_size = 1.0f / static_cast<float>(num_partials * Wt * TILE_WIDTH);
    // The closed sites' reads. Where there are none the accessor aliases the partials, so
    // the kernels' compile-time offsets hold either way and nothing is written through it.
    auto& closed = num_closed_sites > 0 ? tensor_return_value.at(3) : partials;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // A group of sites is one dest session, one accumulator per site, so the group is as
    // large as dest holds. Every group MACs the same candidate tiles, which stay in L1 until
    // the last group is done with them: the sealed set is read once whatever the group count.
    const uint32_t sites_per_group =
        std::min(num_sites, get_dest_reg_count(operation_attributes.compute_kernel_config));
    const uint32_t num_groups = (num_sites + sites_per_group - 1) / sites_per_group;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Plan
    ////////////////////////////////////////////////////////////////////////////
    // Every circular buffer a core holds when its longest run is `max_run` tiles. The split
    // below sizes L1 from this list and the program allocates exactly it, so the two cannot
    // drift apart.
    struct CbSpec {
        CBIndex index;
        uint32_t tiles;
        uint32_t tile_size;
        DataFormat format;
    };
    // Built while the previous one is drained: a group's weights, the shift and mass for the
    // writer, the even groups' partials.
    constexpr uint32_t double_buffered = 2;
    const auto stats_cb = [&](CBIndex index, uint32_t tiles) {
        return CbSpec{index, tiles, stats_tile_size, stats_format};
    };
    const auto value_cb = [&](CBIndex index, uint32_t tiles) {
        return CbSpec{index, tiles, value_tile_size, value_format};
    };
    const auto cb_plan = [&](uint32_t max_run) {
        std::vector<CbSpec> plan = {
            // Scoring, in the statistics tile's own layout: a row is a token, a column a site.
            stats_cb(cb_stats, num_candidates * num_partials),
            stats_cb(cb_rr, num_candidates),
            stats_cb(cb_scores, num_candidates),
            stats_cb(cb_shift, 1),
            stats_cb(cb_summed, num_candidates),
            value_cb(cb_exp, num_candidates),
            // The mixture: a site's weight is column 0 of its own tile, the column a broadcast
            // MAC reads; the reader builds them from the packed exponentials. A run's candidate
            // tiles stay resident for every site group.
            value_cb(cb_values, max_run * num_candidates),
            value_cb(cb_weights, double_buffered * sites_per_group * num_candidates),
            stats_cb(cb_shift_out, double_buffered),
            stats_cb(cb_mass_out, double_buffered),
            // The partials go out on both NoCs, even site groups from the writer and odd ones
            // from the reader. The reader drains an odd group only between building weights, so
            // its buffer holds a whole group: compute must be able to finish one without waiting
            // on it.
            value_cb(cb_partials, double_buffered * sites_per_group),
            value_cb(cb_partials_odd, sites_per_group * max_run),
            stats_cb(cb_staging, num_open_sites),
        };
        if (num_closed_sites > 0) {
            // A closed site's weights are its exponentials over its mass: compute keeps the
            // mass, and turns it into a per-site factor that is 1 / mass in the closed sites'
            // columns and 1 in the rest, through a mask the reader builds once.
            plan.push_back(stats_cb(cb_mass, 1));
            plan.push_back(stats_cb(cb_closed_mask, 1));
            plan.push_back(stats_cb(cb_factor, 1));
        }
        return plan;
    };
    const auto l1_bytes = [](const std::vector<CbSpec>& plan) {
        uint64_t bytes = 0;
        for (const auto& spec : plan) {
            bytes += uint64_t{spec.tiles} * spec.tile_size;
        }
        return bytes;
    };

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // A work item is a run of tiles along one token row-tile. Splitting rows alone would put
    // a few dozen cores behind every write of the R partials; splitting each row into chunks
    // spreads them over the grid. Each core of a row scores that row itself, a few tiles of
    // arithmetic, rather than wait on another core for it.
    //
    // A core's L1 grows with its run, so where the rows alone fill the grid a row is still cut
    // into as many runs as L1 needs; the cores then take the extra items round-robin.
    // An L1 bank spans the circular buffers' region: above the allocator base, below L1_SMALL.
    const uint64_t l1_for_cbs = device->allocator()->get_bank_size(BufferType::L1);
    const auto run_of = [&](uint32_t chunks) { return (Wt + chunks - 1) / chunks; };
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    uint32_t chunks_per_row = std::clamp(grid_cores / Ht, 1u, Wt);
    while (chunks_per_row < Wt && l1_bytes(cb_plan(run_of(chunks_per_row))) > l1_for_cbs) {
        ++chunks_per_row;
    }
    const uint32_t max_run = run_of(chunks_per_row);
    const auto plan = cb_plan(max_run);
    TT_FATAL(
        l1_bytes(plan) <= l1_for_cbs,
        "AttnResBlockMix needs {} B of L1 per core for {} sites over {} candidates; the core has {} B for circular "
        "buffers",
        l1_bytes(plan),
        num_sites,
        num_candidates,
        l1_for_cbs);
    const uint32_t num_items = Ht * chunks_per_row;
    const uint32_t num_cores = std::min(num_items, grid_cores);
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);

    ProgramDescriptor desc;
    for (const auto& spec : plan) {
        desc.cbs.push_back(make_cb(spec.index, all_cores, spec.tiles, spec.tile_size, spec.format));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels
    ////////////////////////////////////////////////////////////////////////////
    // Every kernel takes the shape, the split and the CB ids by name; only the tensor
    // accessors are positional. A core finds its items from its own position, so a launch
    // ships no per-core args. The scalars go as fp32 bit patterns, what the SFPU takes.
    const KernelDescriptor::NamedCompileTimeArgs named_ct_args = {
        {"num_candidates", num_candidates},
        {"num_partials", num_partials},
        {"Ht", Ht},
        {"Wt", Wt},
        {"num_sites", num_sites},
        {"sites_per_group", sites_per_group},
        {"num_groups", num_groups},
        {"num_closed_sites", num_closed_sites},
        {"inv_hidden_size", std::bit_cast<uint32_t>(inv_hidden_size)},
        {"eps", std::bit_cast<uint32_t>(operation_attributes.eps)},
        {"grid_x", grid.x},
        {"num_cores", num_cores},
        {"num_items", num_items},
        {"chunks_per_row", chunks_per_row},
        {"max_run", max_run},
        {"cb_stats", cb_stats},
        {"cb_rr", cb_rr},
        {"cb_scores", cb_scores},
        {"cb_shift", cb_shift},
        {"cb_summed", cb_summed},
        {"cb_exp", cb_exp},
        {"cb_values", cb_values},
        {"cb_weights", cb_weights},
        {"cb_mass", cb_mass},
        {"cb_closed_mask", cb_closed_mask},
        {"cb_factor", cb_factor},
        {"cb_shift_out", cb_shift_out},
        {"cb_mass_out", cb_mass_out},
        {"cb_partials", cb_partials},
        {"cb_partials_odd", cb_partials_odd},
        {"cb_staging", cb_staging},
    };

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*stats.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*first.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*partials.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*closed.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*partials.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*closed.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*shift.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*mass.buffer()).append_to(writer_ct_args);

    // Each candidate is a (snapshot address, first page) pair: the snapshot that holds it and
    // the offset of its plane there. The snapshots share one accessor configuration, which
    // the device operation checks, so the address is all that differs between them.
    KernelDescriptor::RTArgList reader_common;
    reader_common.push_back(stats.buffer());
    reader_common.push_back(partials.buffer());
    reader_common.push_back(closed.buffer());
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
    reader.named_compile_time_args = named_ct_args;
    reader.config = ReaderConfigDescriptor{};
    reader.emplace_common_runtime_args(reader_common);

    KernelDescriptor writer;
    writer.kernel_source = std::string(kKernelDir) + "writer_attn_res_block_mix.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct_args);
    writer.named_compile_time_args = named_ct_args;
    writer.config = WriterConfigDescriptor{};
    writer.emplace_common_runtime_args({partials.buffer(), closed.buffer(), shift.buffer(), mass.buffer()});

    // Scoring runs on the SFPU: every fp32 buffer it reads is only ever copied into dest, so
    // they unpack straight to it rather than being cut to tf32 on the way through srcA.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (const auto index : {cb_stats, cb_rr, cb_scores, cb_shift, cb_summed, cb_mass, cb_closed_mask, cb_factor}) {
        unpack_to_dest_mode[index] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute;
    compute.kernel_source = std::string(kKernelDir) + "attn_res_block_mix.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.named_compile_time_args = named_ct_args;
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
