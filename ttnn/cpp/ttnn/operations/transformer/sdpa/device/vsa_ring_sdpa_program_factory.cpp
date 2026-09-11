// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_program_factory.hpp"

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
    ctx.v_head_offset = tensor_args.vsa.q.logical_shape()[1];  // K/V concatenated on the head dim: V after K
    ctx.gathered_kv = &tensor_args.gathered_kv;
    return ctx;
}

// Re-apply the gathered K/V address (a raw uint32 common arg) on the VSA reader/writer instances. The
// gathered accessor (interleaved DRAM) contributes no common args, so the ring block starts at index 0.
void patch_gathered_address(tt::tt_metal::Program& program, const VsaRingSdpaInputs& tensor_args) {
    const auto addr = static_cast<uint32_t>(tensor_args.gathered_kv.buffer()->address());
    for (uint32_t kernel :
         {kVsaReaderWorkerKernel, kVsaReaderLeaderKernel, kVsaWriterWorkerKernel, kVsaWriterLeaderKernel}) {
        auto& crt = tt::tt_metal::GetCommonRuntimeArgs(program, kernel);
        TT_FATAL(crt.size() > kRingCommonArgGatheredAddr, "vsa_ring_sdpa: ring common args too short");
        crt[kRingCommonArgGatheredAddr] = addr;
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
        // The multi-worker all-gather (the standalone all_gather_async's kernels): its senders forward the
        // concatenated K/V shard around the ring into the gathered buffer (the local shard is not written) and
        // signal the leaders per landed shard with the OpSignaler protocol RingSDPAOpReceiver consumes.
        auto artifacts = ttnn::build_all_gather_async_minimal_default_program_artifacts(
            program,
            tensor_args.vsa.k,  // the local concatenated K/V shard
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
            /*chunks_per_sync=*/std::nullopt,
            args.num_workers_per_link,
            /*num_buffers_per_channel=*/std::nullopt,
            /*core_grid_offset=*/tt::tt_metal::CoreCoord{0, 0},
            /*reverse_order=*/false,
            /*sub_core_grid=*/std::nullopt);
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
        patch_gathered_address(program, tensor_args);
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
