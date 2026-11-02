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
#include "ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"

namespace ttnn::experimental::prim {

namespace {

// The reduce-scatter is the first thing added to each (empty) per-coordinate descriptor, so its reader and writer
// are kernels 0 and 1. A Program built from the descriptor uses the same indices as kernel handles; the cache-hit
// override relies on that to find the semaphore slots. build_program_descriptor() checks it.
constexpr ReduceScatterDescriptorArtifacts kReduceScatterKernels{.reader_kernel_index = 0, .writer_kernel_index = 1};

tt::tt_metal::ProgramDescriptor build_program_descriptor(
    const MatmulReduceScatterAsyncParams& args,
    const ttnn::MeshCoordinate& mesh_coord,
    const MatmulReduceScatterAsyncInputs& tensor_args,
    MatmulReduceScatterAsyncResult& output_tensors) {
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

    tt::tt_metal::ProgramDescriptor desc;

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

    auto resolved_reduce_scatter_compute_kernel_config =
        ttnn::ccl::resolve_fp32_acc_compute_kernel_config(std::nullopt, output_tensors.mm.dtype());

    // Reduce Scatter
    const auto reduce_scatter_kernels = build_ring_reduce_scatter_minimal_async_program_descriptor(
        desc,
        output_tensors.mm,
        tensor_args.persistent_intermediate,
        /*penult_intermediate_tensor=*/std::nullopt,  // contiguous intermediate path not supported through here.
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
        resolved_reduce_scatter_compute_kernel_config);
    TT_FATAL(
        reduce_scatter_kernels.reader_kernel_index == kReduceScatterKernels.reader_kernel_index &&
            reduce_scatter_kernels.writer_kernel_index == kReduceScatterKernels.writer_kernel_index,
        "matmul_reduce_scatter_async: the reduce-scatter kernels must be the first two in the descriptor "
        "(got reader {}, writer {}); the cache-hit override depends on it",
        reduce_scatter_kernels.reader_kernel_index,
        reduce_scatter_kernels.writer_kernel_index);

    // Create a matmul signal info object that gets populated by the matmul kernel
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::REDUCE_SCATTER);

    matmul_fused_op_signaler->init_reduce_scatter(
        reduce_scatter_fused_op_signaler->fused_op_receiver_cores_noc,
        reduce_scatter_fused_op_signaler->fused_op_receiver_signal_semaphores,
        reduce_scatter_fused_op_signaler->fused_op_signaler_mode);

    // Matmul
    ttnn::prim::ccl_fusion::matmul_multi_core_reuse_mcast_2d_optimized_helper(
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

}  // namespace

tt::tt_metal::WorkloadDescriptor MatmulReduceScatterAsyncProgramFactory::create_workload_descriptor(
    const MatmulReduceScatterAsyncParams& args,
    const MatmulReduceScatterAsyncInputs& tensor_args,
    MatmulReduceScatterAsyncResult& output_tensors,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    const auto coords = tensor_coords.coords();
    workload.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord), build_program_descriptor(args, coord, tensor_args, output_tensors)});
    }
    return workload;
}

MatmulReduceScatterAsyncMeshWorkloadFactory::cached_mesh_workload_t
MatmulReduceScatterAsyncMeshWorkloadFactory::create_mesh_workload(
    const MatmulReduceScatterAsyncParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const MatmulReduceScatterAsyncInputs& tensor_args,
    MatmulReduceScatterAsyncResult& output_tensors) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output_tensors);
}

void MatmulReduceScatterAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const MatmulReduceScatterAsyncParams& args,
    const MatmulReduceScatterAsyncInputs& tensor_args,
    MatmulReduceScatterAsyncResult& output_tensors) {
    // Tensor addresses (matmul in0/in1/bias/output, reduce-scatter intermediate/output, tensor-backed CBs).
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output_tensors);

    // The caller-owned GlobalSemaphores.
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        apply_ring_reduce_scatter_semaphore_args(
            program,
            kReduceScatterKernels,
            args.reduce_scatter_params.barrier_semaphore,
            args.reduce_scatter_params.semaphore);
    }
}

}  // namespace ttnn::experimental::prim
