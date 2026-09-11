// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_stream_descriptor.hpp"

namespace ttnn::prim {

namespace {

VsaRingContext build_ring_context(
    const VsaRingSdpaParams& args, const VsaRingSdpaInputs& tensor_args, const ttnn::MeshCoordinate& coord) {
    VsaRingContext ctx;
    ctx.coord = coord;
    ctx.ring_size = static_cast<uint32_t>(args.ag.ring_size);
    ctx.device_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(tensor_args.vsa.q, coord, args.ag.cluster_axis);
    ctx.forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.vsa.q, coord, 1, args.ag.topology, args.ag.cluster_axis);
    ctx.backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.vsa.q, coord, -1, args.ag.topology, args.ag.cluster_axis);
    // Shards each chain delivers: the all-gather helper's own derivation (get_forward_backward_configuration
    // + the even-index swap), consumed by RingSDPAOpReceiver in the same order as ring_joint_sdpa.
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ttnn::ccl::get_forward_backward_configuration(ctx.ring_size, ctx.device_index, args.ag.topology);
    (void)dynamic_alternate;
    TT_FATAL(args.ag.topology == ttnn::ccl::Topology::Ring, "vsa_ring_sdpa: topology must be Ring");
    if (ctx.device_index % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }
    ctx.forward_writes_expected = static_cast<uint32_t>(num_targets_forward);
    ctx.backward_writes_expected = static_cast<uint32_t>(num_targets_backward);
    ctx.ag = &args.ag;
    ctx.gathered_k = &tensor_args.gathered_k;
    ctx.gathered_v = &tensor_args.gathered_v;
    ctx.ccl_core_grid_offset = args.ccl_core_grid_offset;
    return ctx;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor VsaRingSdpaProgramFactory::create_workload_descriptor(
    const VsaRingSdpaParams& args,
    const VsaRingSdpaInputs& tensor_args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        const VsaRingContext ctx = build_ring_context(args, tensor_args, coord);
        auto desc = build_vsa_sdpa_stream_descriptor(args.vsa, tensor_args.vsa, output, &ctx);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

VsaRingSdpaMeshWorkloadFactory::cached_mesh_workload_t VsaRingSdpaMeshWorkloadFactory::create_mesh_workload(
    const VsaRingSdpaParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const VsaRingSdpaInputs& tensor_args,
    Tensor& output) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output);
}

namespace {

void patch_all_gather_semaphores(tt::tt_metal::Program& program, const VsaRingSdpaParams& args) {
    namespace dyn = ttnn::experimental::prim::ring_attention_all_gather_async_dynamic;
    const auto fwd_sem = static_cast<uint32_t>(args.ag.semaphore.at(dyn::kForwardSemaphoreIdx).address());
    const auto bwd_sem = static_cast<uint32_t>(args.ag.semaphore.at(dyn::kBackwardSemaphoreIdx).address());
    // The all-gather kernels are the four pushed after the VSA kernels (handles == push order, see the
    // descriptor header); their semaphore-address slots are the standalone op's kReaderSemaphoreArg /
    // kWriterSemaphoreArg. Forward kernels carry semaphore[kForwardSemaphoreIdx], backward the other.
    const auto metas = tt::tt_metal::detail::collect_kernel_meta(program, nullptr);
    TT_FATAL(
        metas.size() == kVsaStreamKernelCount + 4,
        "vsa_ring_sdpa: expected {} kernels, program has {}",
        kVsaStreamKernelCount + 4,
        metas.size());
    const auto patch = [&](uint32_t kernel_idx, uint32_t slot, uint32_t sem, uint32_t& counter) {
        auto& by_core = tt::tt_metal::GetRuntimeArgs(program, kernel_idx);
        for (auto& by_y : by_core) {
            for (auto& a : by_y) {
                if (a.size() > slot) {
                    a[slot] = sem;
                    ++counter;
                }
            }
        }
    };
    uint32_t patched_readers = 0, patched_writers = 0;
    patch(kVsaStreamKernelCount + dyn::kReaderForwardKernelIdx, dyn::kReaderSemaphoreArg, fwd_sem, patched_readers);
    patch(kVsaStreamKernelCount + dyn::kWriterForwardKernelIdx, dyn::kWriterSemaphoreArg, fwd_sem, patched_writers);
    patch(kVsaStreamKernelCount + dyn::kReaderBackwardKernelIdx, dyn::kReaderSemaphoreArg, bwd_sem, patched_readers);
    patch(kVsaStreamKernelCount + dyn::kWriterBackwardKernelIdx, dyn::kWriterSemaphoreArg, bwd_sem, patched_writers);
    TT_FATAL(
        patched_readers == 2 * args.ag.num_links && patched_writers == 2 * args.ag.num_links,
        "vsa_ring_sdpa: patched {} all-gather readers and {} writers, expected {} each",
        patched_readers,
        patched_writers,
        2 * args.ag.num_links);
}

}  // namespace

void VsaRingSdpaMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const VsaRingSdpaParams& args,
    const VsaRingSdpaInputs& tensor_args,
    Tensor& output) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output);

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        patch_vsa_sdpa_stream_runtime_args(program, args.vsa, tensor_args.vsa, output, /*ring=*/true);
        patch_all_gather_semaphores(program, args);
    }
}

}  // namespace ttnn::prim
