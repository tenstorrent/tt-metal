// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_program_factory.hpp"

#include <array>

#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_stream_descriptor.hpp"

namespace ttnn::prim {

namespace {

// Sender cores the gather places from (0, 0) in row-major order: per link, two directions of (workers + MUX)
// cores. The VSA grid starts below the rows they fill.
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
    ctx.gathered_k = &tensor_args.gathered_k;
    ctx.gathered_v = &tensor_args.gathered_v;
    return ctx;
}

constexpr std::array<uint32_t, 4> kVsaRingKernels = {
    kVsaReaderWorkerKernel, kVsaReaderLeaderKernel, kVsaWriterWorkerKernel, kVsaWriterLeaderKernel};

// The VSA accessors (interleaved DRAM) contribute no common args, so the ring block starts at index 0. Fill the
// packet geometry, the workers' out_ready_sem addresses and the poll table once the gather has placed its cores
// (create); re-apply the raw addresses on cache hits (patch_ring_addresses). The poll table holds each worker's
// row range as tile indices of the token-major sequence (row * 2*H*DHt), which is what RingGate compares.
void fill_ring_poll_table(
    tt::tt_metal::Program& program,
    const VsaRingSdpaParams& args,
    const VsaRingSdpaInputs& tensor_args,
    const VsaKvGatherArtifacts& a) {
    const uint32_t G = a.num_links * a.num_workers_per_direction;
    TT_FATAL(
        a.num_workers_per_direction == args.num_workers_per_link && a.num_links == args.ag.num_links &&
            a.row_ranges.size() == G,
        "vsa_ring_sdpa: gather placement ({} links x {} workers, {} row ranges) differs from the request ({} x {})",
        a.num_links,
        a.num_workers_per_direction,
        a.row_ranges.size(),
        args.ag.num_links,
        args.num_workers_per_link);
    auto* device = tensor_args.vsa.q.device();
    const auto ks = tensor_args.vsa.k.logical_shape();
    const uint32_t seq_row_tiles = 2 * ks[1] * (ks[3] / tt::constants::TILE_WIDTH);  // K heads, then V heads
    for (uint32_t kernel : kVsaRingKernels) {
        auto& crt = tt::tt_metal::GetCommonRuntimeArgs(program, kernel);
        TT_FATAL(
            crt.size() >= kRingCommonArgPollTable + 2 * G * kRingPollWordsPerWorker,
            "vsa_ring_sdpa: ring common args too short ({} words for {} workers)",
            crt.size(),
            G);
        crt[kRingCommonArgTilesPerPacket] = a.tiles_per_packet;
        crt[kRingCommonArgChunksPerSync] = a.chunks_per_sync;
        for (uint32_t d = 0; d < a.num_directions_per_link; ++d) {
            for (uint32_t link = 0; link < a.num_links; ++link) {
                for (uint32_t w = 0; w < a.num_workers_per_direction; ++w) {
                    const auto vc = device->worker_core_from_logical_core(a.worker_core(link, d, w));
                    const uint32_t g = link * a.num_workers_per_direction + w;
                    const uint32_t at = kRingCommonArgPollTable + kRingPollWordsPerWorker * (d * G + g);
                    crt[at] = static_cast<uint32_t>(vc.x) | (static_cast<uint32_t>(vc.y) << 16);
                    crt[at + 1] = a.row_ranges[g].first * seq_row_tiles;
                    crt[at + 2] = a.row_ranges[g].second * seq_row_tiles;
                }
            }
        }
    }
}

void patch_ring_addresses(tt::tt_metal::Program& program, const VsaRingSdpaParams& args, const VsaRingSdpaInputs& t) {
    for (uint32_t kernel : kVsaRingKernels) {
        auto& crt = tt::tt_metal::GetCommonRuntimeArgs(program, kernel);
        TT_FATAL(crt.size() > kRingCommonArgGatheredVAddr, "vsa_ring_sdpa: ring common args too short");
        crt[kRingCommonArgGatheredKAddr] = static_cast<uint32_t>(t.gathered_k.buffer()->address());
        crt[kRingCommonArgGatheredVAddr] = static_cast<uint32_t>(t.gathered_v.buffer()->address());
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
        // VSA kernels first (handles 0..4), then the gather's reader, writer and MUX kernels.
        tt::tt_metal::Program program{build_vsa_sdpa_stream_descriptor(args.vsa, tensor_args.vsa, output, &ctx)};

        ttnn::experimental::ccl::AllGatherFusedOpSignaler signaler;
        signaler.init_fused_op(
            ctx.receiver_cores_noc, ctx.receiver_semaphores, ttnn::experimental::ccl::FusedOpSignalerMode::MULTI);
        const std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.vsa.q, coord, 1, args.ag.topology, args.ag.cluster_axis);
        const std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.vsa.q, coord, -1, args.ag.topology, args.ag.cluster_axis);
        // The op's K/V gather: its senders forward this device's K and V shards around the ring, token-major, into
        // the gathered buffers (the local shard is not written) and signal the leaders per landed shard with the
        // OpSignaler protocol RingSDPAOpReceiver consumes; the leaders also poll the workers' landed-tile counters
        // to gate per block (fill_ring_poll_table).
        auto artifacts = build_vsa_kv_gather(
            program,
            tensor_args.vsa.k,
            tensor_args.vsa.v,
            tensor_args.gathered_k,
            tensor_args.gathered_v,
            coord,
            forward_coord,
            backward_coord,
            args.ag.num_links,
            static_cast<uint32_t>(args.ag.ring_size),
            ctx.device_index,
            args.ag.semaphore,
            args.ag.sub_device_id,
            signaler,
            args.num_workers_per_link,
            /*chunks_per_sync=*/std::nullopt);
        fill_ring_poll_table(program, args, tensor_args, artifacts);
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
        vsa_kv_gather_override_runtime_arguments(
            program,
            a,
            args.ag.semaphore,
            tensor_args.vsa.k,
            tensor_args.vsa.v,
            tensor_args.gathered_k,
            tensor_args.gathered_v);
    }
}

}  // namespace ttnn::prim
