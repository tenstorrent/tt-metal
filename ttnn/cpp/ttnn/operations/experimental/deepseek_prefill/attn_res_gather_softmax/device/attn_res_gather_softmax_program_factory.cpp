// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_gather_softmax_program_factory.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <unordered_set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr auto kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_gather_softmax/device/kernels/";

// The row weights `a` and `b`, and the shift and mass they are derived from.
constexpr uint32_t kRowWeights = 2;
constexpr uint32_t kFixedScalars = 2;
// A rank's contribution to the live score: its sum of squares and its dots.
constexpr uint32_t kStatsPerPartial = 2;
constexpr uint32_t kStatsPerRow = 2;
// Ceiling on the gather's staging buffer, in tiles.
constexpr uint32_t kMaxStageTiles = 64;

// The most cores a token row is split across. Both passes take a core's runs from DRAM
// once and hold them, so more chunks shorten every core's share of the reads; the
// row's leader then sums one partial pair per chunk, which is what bounds it.
constexpr uint32_t kMaxChunksPerRow = 4;
// The common runtime args. Everything a cache hit patches is here, so a hit costs a
// handful of writes per program rather than a handful per core.
enum ReaderCommonArg : uint32_t {
    kReaderSite,
    kReaderPrefixAddr,
    kReaderPartialAddr,
    kReaderQAddr,
    kReaderPendingAddr
};
enum WriterCommonArg : uint32_t {
    kWriterShiftSite,
    kWriterMassSite,
    kWriterStatsAddr,
    kWriterShiftAddr,
    kWriterMassAddr,
    kWriterOutAddr,
    kWriterTotalAddr,
    kWriterRank,
    kWriterGatherX,
    kWriterGatherY,
};
// Where the gather kernel's statistics-tensor address and semaphore address sit.
constexpr uint32_t kGatherStatsAddrIdx = 0;
constexpr uint32_t kGatherSemAddrIdx = 2;

}  // namespace

AttnResGatherSoftmaxSiteOffsets attn_res_gather_softmax_site_offsets(
    const AttnResGatherSoftmaxParams& operation_attributes, const AttnResGatherSoftmaxInputs& tensor_args) {
    const auto& input_shape = tensor_args.partial.padded_shape();
    const uint32_t Wt = input_shape[-1] / TILE_WIDTH;
    const uint32_t Ht = input_shape[-2] / TILE_HEIGHT;

    // A scalar operand is one tile column wide, so its dim-0 plane is Ht pages and
    // selecting a read site is a page offset. Resolved here rather than in the kernel
    // because a shared operand ignores the site, and that test belongs where the
    // shapes are.
    const auto scalar_page_offset = [&](const Tensor& scalar) {
        return scalar.padded_shape()[0] == 1 ? 0u : operation_attributes.site * Ht;
    };

    return {
        .shift = scalar_page_offset(tensor_args.shift),
        .mass = scalar_page_offset(tensor_args.mass),
        .partial = input_shape[0] == 1 ? 0u : operation_attributes.site * Ht * Wt,
    };
}

AttnResGatherSoftmaxMeshWorkloadFactory::cached_program_t AttnResGatherSoftmaxMeshWorkloadFactory::create_at(
    const AttnResGatherSoftmaxParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coord,
    const AttnResGatherSoftmaxInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    ttnn::MeshDevice* mesh_device = tensor_args.partial.device();
    const auto mesh_view = mesh_device->get_view();
    TT_FATAL(mesh_view.is_mesh_2d(), "attn_res_gather_softmax requires a 2D mesh to resolve the cluster axis");

    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t cluster_axis = operation_attributes.cluster_axis;

    const auto fabric_node_ids = (cluster_axis == 0) ? mesh_view.get_fabric_node_ids_on_column(mesh_coord[1])
                                                     : mesh_view.get_fabric_node_ids_on_row(mesh_coord[0]);
    TT_FATAL(
        fabric_node_ids.size() == ring_size,
        "ring_size {} does not match the {} devices on cluster axis {}",
        ring_size,
        fabric_node_ids.size(),
        cluster_axis);

    // This chip's rank names the slot of the statistics tensor it fills. It says nothing
    // about which way a peer lies: the routing tables answer that, and on an axis that
    // wraps they disagree with rank order.
    const uint32_t my_rank = mesh_coord[cluster_axis];

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto& partial = tensor_args.partial;
    const auto& running_sum = tensor_args.running_sum;
    const auto& shift = tensor_args.shift;
    const auto& mass = tensor_args.mass;
    const auto& q = tensor_args.q;
    const auto& stats = tensor_args.stats;
    auto& output = tensor_return_value.at(0);

    // A deferred residual write, settled inside pass one against the very runs it is
    // about to be reduced over. The settled runs stay resident on their core, so the fold
    // in pass two weights them directly.
    const bool fuse_add = tensor_args.pending.has_value();
    const auto& pending = fuse_add ? tensor_args.pending.value() : running_sum;
    auto& total = fuse_add ? tensor_return_value.at(1) : output;

    const auto wide_data_format = datatype_to_dataformat_converter(partial.dtype());
    const auto wide_tile_size = tt::tile_size(wide_data_format);
    const auto scalar_data_format = datatype_to_dataformat_converter(shift.dtype());
    const auto scalar_tile_size = tt::tile_size(scalar_data_format);
    const auto output_data_format = datatype_to_dataformat_converter(output.dtype());
    const auto output_tile_size = tt::tile_size(output_data_format);

    const auto& input_shape = partial.padded_shape();
    const uint32_t Wt = input_shape[-1] / TILE_WIDTH;
    const uint32_t Ht = input_shape[-2] / TILE_HEIGHT;
    const uint32_t kScalars = kFixedScalars + kStatsPerPartial * ring_size;

    const auto site_offsets = attn_res_gather_softmax_site_offsets(operation_attributes, tensor_args);

    // The statistics cross packed by column rather than tile-shaped, so what a packet
    // carries is a page of them — a thousand tokens' worth of one rank's statistic
    // instead of thirty-two. The fabric charges per packet almost regardless of payload,
    // so this is most of what the exchange costs. Kernel-side layout in
    // `kernels/dataflow/attn_res_stats_layout.hpp`.
    // A packed token row is one value per token of a row-tile, so a page holds as many
    // of them as a tile has columns. A worker's scratch has to hold the whole gathered
    // set, which is the larger of the two things it packs.
    const uint32_t packed_row_size = scalar_tile_size / TILE_WIDTH;
    const uint32_t pages_per_plane = (Ht + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t packed_tiles =
        (kStatsPerPartial * ring_size * packed_row_size + scalar_tile_size - 1) / scalar_tile_size;

    const auto max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    TT_FATAL(
        scalar_tile_size <= max_payload,
        "A {} B statistics page exceeds the {} B fabric payload limit; narrow the statistics dtype",
        scalar_tile_size,
        max_payload);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(mesh_device->arch(), operation_attributes.compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                            Core Assignment
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::Program program{};
    const auto grid = mesh_device->compute_with_storage_grid_size();

    // One split for both passes (see `kernels/attn_res_core_runs.hpp`): a unit is a run of
    // one token row, and a core reduces its runs in pass one and folds the same runs in
    // pass two, so the stream is read from DRAM once and held between the passes. A row
    // is cut into as many runs as the grid seats, up to `kMaxChunksPerRow` and only into
    // runs that divide it; where even whole rows outnumber the cores, a core takes several.
    const auto grid_cores =
        corerange_to_cores(CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})), std::nullopt, /*row_wise=*/true);
    TT_FATAL(grid_cores.size() > 1, "attn_res_gather_softmax needs a grid with more than one core");

    // One core is held back for the fabric connection. It cannot double as a worker:
    // a link admits a single sender, and that core's two dataflow RISCs are already
    // the reader and the writer.
    const auto max_cores = static_cast<uint32_t>(grid_cores.size() - 1);
    uint32_t chunks_per_row = 1;
    for (uint32_t c = kMaxChunksPerRow; c > 1; --c) {
        if (Wt % c == 0 && Ht * c <= max_cores) {
            chunks_per_row = c;
            break;
        }
    }
    const uint32_t chunk_tiles = Wt / chunks_per_row;
    const uint32_t num_units = Ht * chunks_per_row;
    const uint32_t units_per_core = (num_units + max_cores - 1) / max_cores;
    const uint32_t num_cores = (num_units + units_per_core - 1) / units_per_core;
    // A split row is one unit per core: a row's leader sums its chunks' pairs as they
    // arrive, which assumes each of them is a core of its own.
    TT_FATAL(chunks_per_row == 1 || units_per_core == 1, "a split row must take one core per chunk");
    const CoreRangeSet unit_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const auto worker_cores = corerange_to_cores(unit_cores, std::nullopt, /*row_wise=*/true);
    const CoreCoord gather_core = grid_cores.back();
    const CoreRangeSet gather_core_set{CoreRange(gather_core, gather_core)};
    TT_FATAL(!unit_cores.contains(gather_core), "the gather core must hold no rows");
    const CoreRangeSet all_cores = unit_cores.merge(gather_core_set);
    // The cores that park a row's statistics and announce it to the gather core: every
    // core where rows are whole, a row's leader where they are split.
    const uint32_t num_parking_cores = chunks_per_row == 1 ? num_cores : Ht;

    ////////////////////////////////////////////////////////////////////////////
    //                            Validation
    ////////////////////////////////////////////////////////////////////////////
    // The fold alternates its srcA operand between the two full-width CBs, which one
    // `bcast_init` configures for both.
    TT_FATAL(
        partial.dtype() == running_sum.dtype(),
        "partial ({}) and running_sum ({}) share one unpacker configuration and must share a dtype",
        partial.dtype(),
        running_sum.dtype());
    // shift, mass and the gathered statistics reach the derivation through one CB.
    TT_FATAL(
        shift.dtype() == mass.dtype() && shift.dtype() == stats.dtype(),
        "shift ({}), mass ({}) and stats ({}) share one circular buffer and must share a dtype",
        shift.dtype(),
        mass.dtype(),
        stats.dtype());
    const auto& stats_shape = stats.padded_shape();
    TT_FATAL(
        stats_shape[1] == kStatsPerPartial * ring_size && stats_shape[-2] == input_shape[-2],
        "stats is {} but the exchange needs [1, {}, {}, 1]: one sum-of-squares and one dots plane per rank",
        stats_shape,
        kStatsPerPartial * ring_size,
        input_shape[-2]);

    // A core's runs stay resident from pass one to the fold: the partial, and the stream
    // they are reduced over, settled or not. The settled stream's second copy and the
    // pending write only ever hold the run in flight.
    const uint32_t resident_tiles = units_per_core * chunk_tiles;
    const uint32_t l1_per_core =
        resident_tiles * wide_tile_size                                         // partial
        + resident_tiles * wide_tile_size                                       // running_sum
        + chunk_tiles * wide_tile_size                                          // q, resident
        + chunk_tiles * wide_tile_size                                          // the reduce's transformed run
        + (fuse_add ? (resident_tiles + 2 * chunk_tiles) * wide_tile_size : 0)  // the sum, its copy, pending
        + scalar_tile_size                                                      // the reduce scaler
        + kScalars * scalar_tile_size                                           // shift, mass, gathered statistics
        + kRowWeights * scalar_tile_size                                        // the row weights
        + kStatsPerRow * (chunks_per_row + 1) * scalar_tile_size                // the run's and row's pairs
        + packed_tiles * scalar_tile_size                                       // the packed statistics
        + 2 * output_tile_size;
    TT_FATAL(
        l1_per_core <= mesh_device->l1_size_per_core(),
        "attn_res_gather_softmax needs {} B of L1 per core at Wt {} but the core has {} B",
        l1_per_core,
        Wt,
        mesh_device->l1_size_per_core());

    ////////////////////////////////////////////////////////////////////////////
    //                      Circular Buffers and Semaphores
    ////////////////////////////////////////////////////////////////////////////
    const auto make_cb =
        [&program](
            uint32_t index, const CoreRangeSet& cores, uint32_t num_tiles, uint32_t tile_size, tt::DataFormat format) {
            tt::tt_metal::CircularBufferConfig config =
                tt::tt_metal::CircularBufferConfig(num_tiles * tile_size, {{index, format}})
                    .set_page_size(index, tile_size);
            return tt::tt_metal::CreateCircularBuffer(program, cores, config);
        };

    make_cb(tt::CBIndex::c_0, unit_cores, resident_tiles, wide_tile_size, wide_data_format);  // partial
    make_cb(tt::CBIndex::c_1, unit_cores, 1, scalar_tile_size, scalar_data_format);           // reduce scaler
    make_cb(tt::CBIndex::c_2, unit_cores, chunk_tiles, wide_tile_size, wide_data_format);     // q
    make_cb(tt::CBIndex::c_3, unit_cores, resident_tiles, wide_tile_size, wide_data_format);  // running_sum
    make_cb(tt::CBIndex::c_4, unit_cores, kRowWeights, scalar_tile_size, scalar_data_format);
    make_cb(tt::CBIndex::c_5, unit_cores, kScalars, scalar_tile_size, scalar_data_format);
    make_cb(tt::CBIndex::c_6, unit_cores, chunk_tiles, wide_tile_size, wide_data_format);  // reduce input
    make_cb(tt::CBIndex::c_7, unit_cores, kStatsPerRow, scalar_tile_size, scalar_data_format);
    make_cb(tt::CBIndex::c_16, unit_cores, 2, output_tile_size, output_data_format);
    // A worker's scratch for the packed statistics: the two columns it parks in pass
    // one, and the whole gathered set it collects in pass two.
    make_cb(tt::CBIndex::c_13, unit_cores, packed_tiles, scalar_tile_size, scalar_data_format);
    if (chunks_per_row > 1) {
        // A row leader's: its other chunks' pairs, written in by them, and the row's sum.
        // Allocated on every core, so the senders find it at their own copy's address.
        make_cb(
            tt::CBIndex::c_14, unit_cores, kStatsPerRow * (chunks_per_row - 1), scalar_tile_size, scalar_data_format);
        make_cb(tt::CBIndex::c_15, unit_cores, kStatsPerRow, scalar_tile_size, scalar_data_format);
    }

    if (fuse_add) {
        make_cb(tt::CBIndex::c_10, unit_cores, chunk_tiles, wide_tile_size, wide_data_format);  // pending
        // The settled run, packed twice from one dest register: a circular buffer has a
        // single consumer. One copy stays resident for the reduces and the fold; the
        // writer parks the other in DRAM as the stream the caller carries forward.
        make_cb(tt::CBIndex::c_11, unit_cores, resident_tiles, wide_tile_size, wide_data_format);
        make_cb(tt::CBIndex::c_12, unit_cores, chunk_tiles, wide_tile_size, wide_data_format);
    }

    // The gather core holds no operand — one route header and one increment header per
    // peer, and a staging buffer for the plane it sends.
    const uint32_t kPeers = ring_size - 1;
    const auto packet_header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    make_cb(tt::CBIndex::c_8, gather_core_set, kStatsPerRow * kPeers, packet_header_size, tt::DataFormat::UInt32);

    // The gather stages a run of pages and sends them behind a single read barrier.
    // A barrier per page would put the exchange on the DRAM latency ladder — one round
    // trip per page, serialized — at exactly the moment every fold core is prefetching
    // against the same DRAM. The cap bounds the buffer for sequences long enough that
    // the whole plane would not fit L1, which the packed layout puts far out of reach:
    // a page carries a thousand tokens.
    const uint32_t stage_tiles = std::min(kStatsPerRow * pages_per_plane, kMaxStageTiles);
    make_cb(tt::CBIndex::c_9, gather_core_set, stage_tiles, scalar_tile_size, scalar_data_format);

    // Local counters, so a program-launch reset is what we want: the workers that
    // increment `ready` and the gather core that increments `done` are all in this
    // program on this chip.
    const auto ready_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    const auto done_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    // A row leader's count of its other chunks' pairs.
    const auto chunk_sem_id = tt::tt_metal::CreateSemaphore(program, unit_cores, 0);

    ////////////////////////////////////////////////////////////////////////////
    //                              Kernels
    ////////////////////////////////////////////////////////////////////////////
    // Each core derives its units from its own position (see
    // `kernels/attn_res_core_runs.hpp`), so the split rides in compile-time args and a
    // launch ships no per-core runtime args at all.
    const std::array<uint32_t, 5> split_ct_args = {grid.x, units_per_core, num_units, chunks_per_row, chunk_tiles};

    // `pending` aliases running_sum where no write was handed in, so the accessor list is
    // the same length either way and the kernel's compile-time offsets do not move. The
    // kernel never reads through it there.
    std::vector<uint32_t> reader_ct_args = {Wt, static_cast<uint32_t>(fuse_add)};
    TensorAccessorArgs(*running_sum.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*partial.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*q.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*pending.buffer()).append_to(reader_ct_args);
    reader_ct_args.insert(reader_ct_args.end(), split_ct_args.begin(), split_ct_args.end());

    const auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + "dataflow/reader_attn_res_gather_softmax.cpp",
        unit_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> writer_ct_args = {
        Wt, ring_size, Ht, ready_sem_id, done_sem_id, static_cast<uint32_t>(fuse_add), chunk_sem_id};
    TensorAccessorArgs(*stats.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*shift.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*mass.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*total.buffer()).append_to(writer_ct_args);
    writer_ct_args.insert(writer_ct_args.end(), split_ct_args.begin(), split_ct_args.end());

    const auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + "dataflow/writer_attn_res_gather_softmax.cpp",
        unit_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    std::vector<uint32_t> compute_ct_args = {
        Wt,
        ring_size,
        std::bit_cast<uint32_t>(operation_attributes.inv_hidden_size),
        std::bit_cast<uint32_t>(operation_attributes.eps),
        static_cast<uint32_t>(fuse_add)};
    compute_ct_args.insert(compute_ct_args.end(), split_ct_args.begin(), split_ct_args.end());

    tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + "compute/attn_res_gather_softmax.cpp",
        unit_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_ct_args});

    std::vector<uint32_t> gather_ct_args = {ring_size, Ht, num_parking_cores, ready_sem_id, done_sem_id, stage_tiles};
    TensorAccessorArgs(*stats.buffer()).append_to(gather_ct_args);

    auto gather_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + "dataflow/gather_attn_res_gather_softmax.cpp",
        gather_core_set,
        tt::tt_metal::ReaderDataMovementConfig(gather_ct_args));

    ////////////////////////////////////////////////////////////////////////////
    //                           Runtime Arguments
    ////////////////////////////////////////////////////////////////////////////
    const auto gather_physical = mesh_device->worker_core_from_logical_core(gather_core);
    tt::tt_metal::SetCommonRuntimeArgs(
        program,
        reader_kernel_id,
        {site_offsets.partial,
         running_sum.buffer()->address(),
         partial.buffer()->address(),
         q.buffer()->address(),
         pending.buffer()->address()});
    std::vector<uint32_t> writer_common = {
        site_offsets.shift,
        site_offsets.mass,
        stats.buffer()->address(),
        shift.buffer()->address(),
        mass.buffer()->address(),
        output.buffer()->address(),
        total.buffer()->address(),
        my_rank,
        gather_physical.x,
        gather_physical.y};
    if (chunks_per_row > 1) {
        // Each row's leader, its chunk 0 core, where the row's other chunks send their pairs.
        for (uint32_t g = 0; g < Ht; ++g) {
            const auto leader = mesh_device->worker_core_from_logical_core(worker_cores.at(g * chunks_per_row));
            writer_common.push_back((leader.x << 16) | leader.y);
        }
    }
    tt::tt_metal::SetCommonRuntimeArgs(program, writer_kernel_id, writer_common);

    // The closed form the kernels use, checked against the cores it stands in for.
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = worker_cores.at(i);
        TT_FATAL(
            core.y * grid.x + core.x == i,
            "core {} (#{}) does not take the units its position implies; the kernels would read the wrong tiles",
            core,
            i);
    }

    // The release is a multicast per rectangle of the fold set rather than one
    // increment per core, so the fold's cost in cores does not become a cost in
    // wake-up latency. The gather core sits outside every rectangle by construction,
    // so it is not among the destinations it counts.
    std::vector<uint32_t> release_args;
    const auto fold_ranges = unit_cores.merge_ranges().ranges();
    for (const auto& range : fold_ranges) {
        const auto start_physical = mesh_device->worker_core_from_logical_core(range.start_coord);
        const auto end_physical = mesh_device->worker_core_from_logical_core(range.end_coord);
        release_args.push_back(start_physical.x);
        release_args.push_back(start_physical.y);
        release_args.push_back(end_physical.x);
        release_args.push_back(end_physical.y);
        release_args.push_back(static_cast<uint32_t>(range.size()));
    }

    std::vector<uint32_t> gather_rt_args = {
        stats.buffer()->address(),
        my_rank,
        static_cast<uint32_t>(operation_attributes.semaphore.address()),
        static_cast<uint32_t>(fold_ranges.size())};
    gather_rt_args.insert(gather_rt_args.end(), release_args.begin(), release_args.end());

    const auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coord);

    // A peer's route, in the order the gather kernel takes its peers: the peer's node
    // outright, so the connection above only chooses the first hop. The fabric config
    // also decides the packet header type the kernel reads these back with, so host and
    // device stay in step without a flag — and only a 2D config pairs with the header
    // this kernel is compiled against.
    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    TT_FATAL(
        tt::tt_fabric::is_2d_fabric_config(fabric_config),
        "attn_res_gather_softmax needs a 2D fabric, got {}",
        fabric_config);

    // One connection per distinct first-hop direction the peers need, and the routing
    // tables name that direction. Rank order does not: on an axis that wraps, the shorter
    // way round to a higher rank runs in the lower direction, and a connection opened
    // against a peer's own route cannot carry its packets, so that peer waits forever on
    // an arrival nothing will deliver. Peers sharing a first hop share the connection, so
    // an axis of any width needs at most one connection per direction.
    std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
    std::vector<uint32_t> per_connection_links;
    std::unordered_set<tt::tt_fabric::eth_chan_directions> used_directions;
    for (uint32_t p = 0; p < ring_size; ++p) {
        if (p == my_rank) {
            continue;
        }
        const auto peer_node = fabric_node_ids.at(p);
        const auto direction = tt::tt_fabric::get_eth_forwarding_direction(src_fabric_node_id, peer_node);
        TT_FATAL(
            direction.has_value(),
            "No 2D fabric forwarding direction from attn_res_gather_softmax source {} to peer {}",
            src_fabric_node_id,
            peer_node);
        if (!used_directions.insert(direction.value()).second) {
            continue;
        }
        const auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh_device, src_fabric_node_id, peer_node);
        TT_FATAL(
            !links.empty(),
            "No forwarding links from attn_res_gather_softmax source {} to peer {}",
            src_fabric_node_id,
            peer_node);
        dst_nodes.push_back(peer_node);
        // One gather core per chip, so it takes the first link rather than striping.
        per_connection_links.push_back(links.front());
    }

    // The count first: the kernel reads it before building the manager from what follows.
    gather_rt_args.push_back(static_cast<uint32_t>(dst_nodes.size()));
    tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
        src_fabric_node_id, dst_nodes, per_connection_links, program, gather_kernel_id, gather_core, gather_rt_args);

    for (uint32_t p = 0; p < ring_size; ++p) {
        if (p == my_rank) {
            continue;
        }
        gather_rt_args.push_back(*fabric_node_ids.at(p).mesh_id);
        gather_rt_args.push_back(fabric_node_ids.at(p).chip_id);
    }
    tt::tt_metal::SetRuntimeArgs(program, gather_kernel_id, gather_core, gather_rt_args);

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .gather_kernel_id = gather_kernel_id,
         .gather_core = gather_core}};
}

AttnResGatherSoftmaxMeshWorkloadFactory::cached_mesh_workload_t
AttnResGatherSoftmaxMeshWorkloadFactory::create_mesh_workload(
    const AttnResGatherSoftmaxParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AttnResGatherSoftmaxInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto cached_program = create_at(operation_attributes, mesh_coord, tensor_args, tensor_return_value);
            shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
            mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        }
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void AttnResGatherSoftmaxMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AttnResGatherSoftmaxParams& operation_attributes,
    const AttnResGatherSoftmaxInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    const auto site_offsets = attn_res_gather_softmax_site_offsets(operation_attributes, tensor_args);
    const bool fuse_add = tensor_args.pending.has_value();
    const auto pending_addr =
        fuse_add ? tensor_args.pending->buffer()->address() : tensor_args.running_sum.buffer()->address();
    const auto total_addr =
        fuse_add ? tensor_return_value.at(1).buffer()->address() : tensor_return_value.at(0).buffer()->address();

    for (auto& [range, shared_vars] : cached_workload.shared_variables) {
        auto& program = cached_workload.workload.get_programs().at(range);

        // The site is a page offset, so a new site is a common-arg patch rather than a
        // new program. That is what keeps one cached program serving every read site.
        auto& reader_common = tt::tt_metal::GetCommonRuntimeArgs(program, shared_vars.reader_kernel_id);
        reader_common[kReaderSite] = site_offsets.partial;
        reader_common[kReaderPrefixAddr] = tensor_args.running_sum.buffer()->address();
        reader_common[kReaderPartialAddr] = tensor_args.partial.buffer()->address();
        reader_common[kReaderQAddr] = tensor_args.q.buffer()->address();
        reader_common[kReaderPendingAddr] = pending_addr;

        auto& writer_common = tt::tt_metal::GetCommonRuntimeArgs(program, shared_vars.writer_kernel_id);
        writer_common[kWriterShiftSite] = site_offsets.shift;
        writer_common[kWriterMassSite] = site_offsets.mass;
        writer_common[kWriterStatsAddr] = tensor_args.stats.buffer()->address();
        writer_common[kWriterShiftAddr] = tensor_args.shift.buffer()->address();
        writer_common[kWriterMassAddr] = tensor_args.mass.buffer()->address();
        writer_common[kWriterOutAddr] = tensor_return_value.at(0).buffer()->address();
        writer_common[kWriterTotalAddr] = total_addr;

        auto& gather_args_by_core = GetRuntimeArgs(program, shared_vars.gather_kernel_id);
        auto& gather_args = gather_args_by_core[shared_vars.gather_core.x][shared_vars.gather_core.y];
        gather_args[kGatherStatsAddrIdx] = tensor_args.stats.buffer()->address();
        gather_args[kGatherSemAddrIdx] = operation_attributes.semaphore.address();
    }
}

}  // namespace ttnn::experimental::prim
