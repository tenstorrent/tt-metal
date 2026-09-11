// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_program_factory.hpp"

#include <array>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_stream_descriptor.hpp"

namespace ttnn::prim {

namespace {

// Sender cores the multi-worker all-gather places from (0, 0) in row-major order: per link, two directions of
// (workers + 1 MUX) cores. The VSA grid starts below the rows they fill.
uint32_t sender_rows_for(const VsaRingSdpaParams& args, const tt::tt_metal::CoreCoord& grid) {
    const uint32_t mux = args.num_workers_per_link == 1 ? 0u : 1u;
    const uint32_t senders = args.ag.num_links * 2 * (args.num_workers_per_link + mux);
    return (senders + grid.x - 1) / grid.x;
}

VsaRingContext build_ring_context(
    const VsaRingSdpaParams& args, const VsaRingSdpaInputs& tensor_args, const ttnn::MeshCoordinate& coord) {
    VsaRingContext ctx;
    ctx.ring_size = static_cast<uint32_t>(args.ag.ring_size);
    ctx.device_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(tensor_args.vsa.q, coord, args.ag.cluster_axis);
    // Shards each chain delivers: the all-gather's own derivation (get_forward_backward_configuration + the
    // even-index swap) consumed by RingSDPAOpReceiver in the same order as ring_joint_sdpa.
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ttnn::ccl::get_forward_backward_configuration(ctx.ring_size, ctx.device_index, args.ag.topology);
    (void)dynamic_alternate;
    TT_FATAL(args.ag.topology == ttnn::ccl::Topology::Ring, "vsa_ring_sdpa: topology must be Ring");
    if (ctx.device_index % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }
    ctx.forward_writes_expected = static_cast<uint32_t>(num_targets_forward);
    ctx.backward_writes_expected = static_cast<uint32_t>(num_targets_backward);
    ctx.sender_rows = sender_rows_for(args, tensor_args.vsa.q.device()->compute_with_storage_grid_size());
    ctx.workers_per_direction = args.ag.num_links * args.num_workers_per_link;
    ctx.gathered_kv = &tensor_args.gathered_kv;
    return ctx;
}

// The all-gather's tile split and packet geometry, replicated for the leaders' landing gate (RingGate in the
// reader): worker g (global id link * workers + worker, the same for both directions) forwards tiles
// [first_g, end_g) of every slice in packets of `tpp` tiles, and the sender bumps its out_ready_sem every `cps`
// packets. cps is the all-gather's own default for the largest range, passed to the builder explicitly so every
// worker and the leaders agree.
struct PollGeometry {
    uint32_t tpp = 1, cps = 1, G = 1;
    std::vector<std::pair<uint32_t, uint32_t>> ranges;  // [first, end) tiles per worker
};

PollGeometry poll_geometry(const VsaRingSdpaParams& args, const Tensor& kv) {
    const auto shape = kv.logical_shape();
    const uint32_t pages = (shape[2] / tt::constants::TILE_HEIGHT) * (shape[3] / tt::constants::TILE_WIDTH);
    const uint32_t page_bytes = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(kv.dtype()));
    const uint32_t packet_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    PollGeometry g;
    g.tpp = std::min<uint32_t>(4, std::max<uint32_t>(1, packet_bytes / page_bytes));  // scatter write: <= 4 pages
    g.G = args.ag.num_links * args.num_workers_per_link;
    const uint32_t base = pages / g.G, rem = pages % g.G;
    for (uint32_t w = 0; w < g.G; ++w) {
        g.ranges.emplace_back(w * base + std::min(w, rem), (w + 1) * base + std::min(w + 1, rem));
    }
    constexpr uint32_t kHeuristicMaxChunksPerSync = 160;  // the all-gather's HEURISTIC_MAX_CHUNKS_PER_SYNC
    const uint32_t range0 = g.ranges[0].second - g.ranges[0].first;
    g.cps = std::min(std::max<uint32_t>(range0 / g.tpp, 1), kHeuristicMaxChunksPerSync);
    return g;
}

constexpr std::array<uint32_t, 4> kVsaRingKernels = {
    kVsaReaderWorkerKernel, kVsaReaderLeaderKernel, kVsaWriterWorkerKernel, kVsaWriterLeaderKernel};

// The VSA accessors (interleaved DRAM) contribute no common args, so the ring block starts at index 0.
// Fill the packet geometry, the workers' out_ready_sem addresses and the poll table once the all-gather has
// placed its cores (create), and re-apply the raw addresses on cache hits (override).
void fill_ring_poll_table(
    tt::tt_metal::Program& program,
    const VsaRingSdpaParams& args,
    const VsaRingSdpaInputs& tensor_args,
    const ttnn::AllGatherProgramArtifacts& a,
    const PollGeometry& geo) {
    TT_FATAL(
        a.num_workers_per_direction == args.num_workers_per_link && a.num_directions_per_link == 2 &&
            a.all_cores.size() == args.ag.num_links * a.num_cores_per_link,
        "vsa_ring_sdpa: all-gather placement ({} cores, {} workers/direction, {} directions) differs from the "
        "requested {} links x {} workers",
        a.all_cores.size(),
        a.num_workers_per_direction,
        a.num_directions_per_link,
        args.ag.num_links,
        args.num_workers_per_link);
    auto* device = tensor_args.vsa.q.device();
    const uint32_t W = a.num_workers_per_direction, mux = a.num_mux_cores_per_direction_per_link;
    for (uint32_t kernel : kVsaRingKernels) {
        auto& crt = tt::tt_metal::GetCommonRuntimeArgs(program, kernel);
        TT_FATAL(
            crt.size() >= kRingCommonArgPollTable + 2 * geo.G * kRingPollWordsPerWorker,
            "vsa_ring_sdpa: ring common args too short ({} words for {} workers)",
            crt.size(),
            geo.G);
        crt[kRingCommonArgTilesPerPacket] = geo.tpp;
        crt[kRingCommonArgChunksPerSync] = geo.cps;
        for (uint32_t d = 0; d < 2; ++d) {
            for (uint32_t link = 0; link < args.ag.num_links; ++link) {
                for (uint32_t w = 0; w < W; ++w) {
                    // the all-gather's core order: link, direction, then the MUX core(s) and the workers
                    const auto& core = a.all_cores.at(link * a.num_cores_per_link + d * (mux + W) + mux + w);
                    const auto v = device->worker_core_from_logical_core(core);
                    const uint32_t g = link * W + w;
                    const uint32_t at = kRingCommonArgPollTable + kRingPollWordsPerWorker * (d * geo.G + g);
                    crt[at] = static_cast<uint32_t>(v.x) | (static_cast<uint32_t>(v.y) << 16);
                    crt[at + 1] = geo.ranges[g].first;
                    crt[at + 2] = geo.ranges[g].second;
                }
            }
        }
    }
}

void patch_ring_addresses(tt::tt_metal::Program& program, const VsaRingSdpaParams& args, const VsaRingSdpaInputs& t) {
    for (uint32_t kernel : kVsaRingKernels) {
        auto& crt = tt::tt_metal::GetCommonRuntimeArgs(program, kernel);
        TT_FATAL(crt.size() > kRingCommonArgSemAddr1, "vsa_ring_sdpa: ring common args too short");
        crt[kRingCommonArgGatheredAddr] = static_cast<uint32_t>(t.gathered_kv.buffer()->address());
        crt[kRingCommonArgSemAddr0] = static_cast<uint32_t>(args.ag.semaphore.at(0).address());
        crt[kRingCommonArgSemAddr1] = static_cast<uint32_t>(args.ag.semaphore.at(1).address());
    }
}

}  // namespace

VsaRingSdpaMeshWorkloadFactory::cached_mesh_workload_t VsaRingSdpaMeshWorkloadFactory::create_mesh_workload(
    const VsaRingSdpaParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const VsaRingSdpaInputs& tensor_args,
    Tensor& output) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        VsaRingContext ctx = build_ring_context(args, tensor_args, coord);
        // VSA kernels first (handles 0..4), then the all-gather's reader and writer.
        tt::tt_metal::Program program{build_vsa_sdpa_stream_descriptor(args.vsa, tensor_args.vsa, output, &ctx)};

        std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> ag_signaler =
            ttnn::experimental::ccl::AllGatherFusedOpSignaler();
        ag_signaler->init_fused_op(
            ctx.receiver_cores_noc, ctx.receiver_semaphores, ttnn::experimental::ccl::FusedOpSignalerMode::MULTI);
        const std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.vsa.q, coord, 1, args.ag.topology, args.ag.cluster_axis);
        const std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.vsa.q, coord, -1, args.ag.topology, args.ag.cluster_axis);
        Tensor gathered = tensor_args.gathered_kv;  // the builder takes a mutable output tensor
        const PollGeometry geo = poll_geometry(args, tensor_args.vsa.k);
        // The multi-worker all-gather (the standalone all_gather_async's kernels): its senders forward the flat
        // K|V shard around the ring into the gathered buffer (the local shard is not written) and signal the
        // leaders per landed shard with the OpSignaler protocol RingSDPAOpReceiver consumes; the leaders also
        // poll the workers' landed-tile counters to gate per block (fill_ring_poll_table).
        auto artifacts = ttnn::build_all_gather_async_minimal_default_program_artifacts(
            program,
            tensor_args.vsa.k,  // the local flat K|V shard
            coord,
            forward_coord,
            backward_coord,
            gathered,
            /*dim=*/2,
            args.ag.num_links,
            static_cast<uint32_t>(args.ag.ring_size),
            ctx.device_index,
            args.ag.topology,
            args.ag.semaphore,
            /*barrier_semaphore=*/std::nullopt,
            /*using_persistent_buffers=*/true,
            args.ag.sub_device_id,
            ag_signaler,
            /*chunks_per_sync=*/geo.cps,
            args.num_workers_per_link,
            /*num_buffers_per_channel=*/std::nullopt,
            /*core_grid_offset=*/tt::tt_metal::CoreCoord{0, 0},
            /*reverse_order=*/false,
            /*sub_core_grid=*/std::nullopt);
        fill_ring_poll_table(program, args, tensor_args, artifacts, geo);
        patch_ring_addresses(program, args, tensor_args);
        const ttnn::MeshCoordinateRange range(coord, coord);
        workload.add_program(range, std::move(program));
        shared_variables[range] = std::move(artifacts);
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void VsaRingSdpaMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const VsaRingSdpaParams& args,
    const VsaRingSdpaInputs& tensor_args,
    Tensor& output) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& a = cached_workload.shared_variables.at(range);
        patch_vsa_sdpa_stream_runtime_args(program, args.vsa, tensor_args.vsa, output, /*ring=*/true);
        patch_ring_addresses(program, args, tensor_args);
        Tensor gathered = tensor_args.gathered_kv;
        ttnn::all_gather_async_minimal_default_helper_override_runtime_arguments(
            program,
            a.reader_kernel_id,
            a.writer_kernel_id,
            a.all_cores,
            args.ag.num_links,
            a.num_directions_per_link,
            a.num_workers_per_direction,
            a.num_mux_cores_per_direction_per_link,
            a.num_cores_per_link,
            /*barrier_semaphore=*/std::nullopt,
            args.ag.semaphore,
            tensor_args.vsa.k,
            gathered);
    }
}

}  // namespace ttnn::prim
