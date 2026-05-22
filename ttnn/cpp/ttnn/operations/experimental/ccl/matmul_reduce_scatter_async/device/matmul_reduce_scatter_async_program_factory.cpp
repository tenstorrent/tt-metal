// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"

namespace ttnn::experimental::prim {

// Build the per-coord ProgramDescriptor for the fused Matmul + ReduceScatter.
// Mirrors the legacy create_at() body 1:1 but routes through the
// ProgramDescriptor variants of the ring-reduce-scatter and matmul-2D-mcast
// builders.  Both helpers append onto the same `desc` (CBs, kernels,
// semaphores).  The fused-op signaler bridge between the two halves is
// identical to the legacy path.
static tt::tt_metal::ProgramDescriptor build_descriptor_at(
    const MatmulReduceScatterAsyncParams& args,
    const ttnn::MeshCoordinate& mesh_coord,
    const MatmulReduceScatterAsyncInputs& tensor_args,
    MatmulReduceScatterAsyncResult& output_tensors) {
    tt::tt_metal::ProgramDescriptor desc;

    ttnn::ccl::Topology topology = args.reduce_scatter_params.topology;

    const auto& dim = args.reduce_scatter_params.dim;
    const auto& num_links = args.reduce_scatter_params.num_links;
    const auto& ring_size = args.reduce_scatter_params.ring_size;
    const auto& semaphore = args.reduce_scatter_params.semaphore;
    const auto& barrier_semaphore = args.reduce_scatter_params.barrier_semaphore;
    const auto& using_persistent_buffers = args.reduce_scatter_params.using_persistent_buffers;
    const auto& sub_device_id = args.reduce_scatter_params.sub_device_id;

    const auto& program_config = args.matmul_struct.program_config.value();
    auto compute_kernel_config = args.matmul_struct.compute_kernel_config.value();
    bool bcast_batch = args.matmul_struct.bcast_batch.value();
    bool untilize_out = args.matmul_struct.untilize_out;

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input, mesh_coord, 1, args.reduce_scatter_params.topology, args.reduce_scatter_params.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input,
        mesh_coord,
        -1,
        args.reduce_scatter_params.topology,
        args.reduce_scatter_params.cluster_axis);

    uint32_t ring_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input, mesh_coord, args.reduce_scatter_params.cluster_axis);

    // Create the reduce scatter fused op signaler
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> reduce_scatter_fused_op_signaler =
        ttnn::experimental::ccl::ReduceScatterFusedOpSignaler();
    reduce_scatter_fused_op_signaler->init_fused_op();

    // Reduce Scatter - descriptor builder appends kernels/CBs/semaphores onto desc.
    (void)ttnn::experimental::prim::build_ring_reduce_scatter_minimal_async_program_artifacts_descriptor(
        desc,
        output_tensors.mm,
        tensor_args.persistent_intermediate,
        mesh_coord,
        forward_coord,
        backward_coord,
        output_tensors.reduce_scatter,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        reduce_scatter_fused_op_signaler,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        args.reduce_scatter_core_grid_offset,
        std::nullopt);

    // Create a matmul signal info object that gets populated by the matmul kernel
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::REDUCE_SCATTER);

    matmul_fused_op_signaler->init_reduce_scatter(
        reduce_scatter_fused_op_signaler->fused_op_receiver_cores_noc,
        reduce_scatter_fused_op_signaler->fused_op_receiver_signal_semaphores,
        reduce_scatter_fused_op_signaler->fused_op_signaler_mode);

    // Matmul - descriptor builder also appends to the same desc.
    ttnn::prim::matmul_multi_core_reuse_mcast_2d_optimized_helper_descriptor(
        desc,
        tensor_args.input,
        tensor_args.weight,
        tensor_args.bias,
        output_tensors.mm,
        bcast_batch,
        compute_kernel_config,
        program_config,
        untilize_out,
        matmul_fused_op_signaler);

    return desc;
}

tt::tt_metal::WorkloadDescriptor MatmulReduceScatterAsyncProgramFactory::create_workload_descriptor(
    const MatmulReduceScatterAsyncParams& args,
    const MatmulReduceScatterAsyncInputs& tensor_args,
    MatmulReduceScatterAsyncResult& output_tensors,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_descriptor_at(args, coord, tensor_args, output_tensors);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return workload_descriptor;
}

}  // namespace ttnn::experimental::prim
