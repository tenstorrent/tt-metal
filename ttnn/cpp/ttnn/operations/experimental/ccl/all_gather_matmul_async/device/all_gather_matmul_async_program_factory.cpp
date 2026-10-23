// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.hpp"
#include "ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_1d.hpp"
#include "ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/overloaded.hpp>

namespace ttnn::experimental::prim {

namespace {

// The all-gather is the first descriptor in the merge below, so its reader and writer are kernels 0 and 1 of every
// per-coordinate program. A Program built from the descriptor uses the same indices as kernel handles; the cache-hit
// override relies on that to find the semaphore slots. build_program_descriptor() checks it.
constexpr AllGatherDescriptorArtifacts kAllGatherKernels{.reader_kernel_index = 0, .writer_kernel_index = 1};

// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
tt::tt_metal::ProgramDescriptor build_program_descriptor(
    const AllGatherMatmulAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coord,
    const AllGatherMatmulAsyncInputs& tensor_args,
    AllGatherMatmulAsyncResult& tensor_return_value) {
    const auto& ag = operation_attributes.all_gather_async_attributes;
    const Tensor& input_tensor = tensor_args.input_tensor;
    Tensor& all_gather_output_tensor = tensor_return_value[0];
    const Tensor& weight_tensor = tensor_args.weight_tensor;
    Tensor& matmul_output_tensor = tensor_return_value[1];

    const uint32_t ring_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, mesh_coord, ag.cluster_axis);
    const std::optional<MeshCoordinate> forward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, mesh_coord, 1, ag.topology, ag.cluster_axis);
    const std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coord, -1, ag.topology, ag.cluster_axis);

    const bool bcast_batch = operation_attributes.matmul.bcast_batch.value();
    const auto compute_kernel_config = operation_attributes.matmul.compute_kernel_config.value();
    const auto& program_config = operation_attributes.matmul.program_config.value();
    const bool untilize_out = operation_attributes.matmul.untilize_out;

    ////////////// Params for fused op signalers //////////////
    auto tensor_slicer =
        ttnn::ccl::InterleavedRingAllGatherTensorSlicer(input_tensor, all_gather_output_tensor, ag.dim, ring_index);
    bool is_clockwise_direction = true;
    const uint32_t num_transfers = 4;
    const uint32_t weight_tensor_width = weight_tensor.padded_shape()[3] / 32;

    ////////////////////////////////////////////////////////

    // Create a matmul signal info object that gets populated by the matmul kernel
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(ttnn::experimental::ccl::MatmulFusedOpSignalerType::ALL_GATHER);
    matmul_fused_op_signaler->init_all_gather(
        num_transfers,
        ag.ring_size,
        ring_index,
        tensor_slicer.num_cols,
        tensor_slicer.output_page_offset,
        is_clockwise_direction,
        tensor_slicer.num_cols *
            weight_tensor_width /* weight_output_page_offset: stride across a tensor slice in the weight_tensor */
    );

    // Matmul (first, as before: the all-gather needs the matmul signaler's receiver cores and semaphores)
    tt::tt_metal::ProgramDescriptor matmul_desc;
    std::visit(
        ttsl::overloaded{
            [&](const operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& config) {
                ttnn::prim::ccl_fusion::matmul_multi_core_reuse_mcast_2d_optimized_helper(
                    matmul_desc,
                    all_gather_output_tensor,
                    weight_tensor,
                    tensor_args.bias,
                    matmul_output_tensor,
                    bcast_batch,
                    compute_kernel_config,
                    config,
                    untilize_out,
                    matmul_fused_op_signaler);
            },
            [&](const operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& config) {
                ttnn::prim::ccl_fusion::matmul_multi_core_reuse_mcast_1d_optimized_helper(
                    matmul_desc,
                    all_gather_output_tensor,
                    {weight_tensor},
                    tensor_args.bias,
                    {matmul_output_tensor},
                    bcast_batch,
                    compute_kernel_config,
                    config,
                    untilize_out,
                    matmul_fused_op_signaler);
            },
            [&](const auto& /*config*/) {
                TT_THROW("Unsupported MatmulProgramConfig type. Needs to be 1D or 2D Multicast.");
            }},
        program_config);

    // Create the all gather fused op signaler
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::AllGatherFusedOpSignaler();
    all_gather_fused_op_signaler->init_fused_op(
        matmul_fused_op_signaler->fused_op_receiver_cores_noc,
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores,
        matmul_fused_op_signaler->fused_op_signaler_mode);

    // All Gather
    tt::tt_metal::ProgramDescriptor all_gather_desc;
    const auto all_gather_kernels = ttnn::build_all_gather_async_minimal_default_program_descriptor(
        all_gather_desc,
        input_tensor,
        mesh_coord,
        forward_coord,
        backward_coord,
        all_gather_output_tensor,
        ag.dim,
        ag.num_links,
        ag.ring_size,
        ring_index,
        ag.topology,
        ag.semaphore,
        ag.barrier_semaphore,
        ag.using_persistent_buffers,
        ag.sub_device_id,
        all_gather_fused_op_signaler,
        ag.chunks_per_sync,
        ag.num_workers_per_link,
        ag.num_buffers_per_channel,
        operation_attributes.all_gather_core_grid_offset,
        false,  // reverse_order = false by default
        std::nullopt);
    TT_FATAL(
        all_gather_kernels.reader_kernel_index == kAllGatherKernels.reader_kernel_index &&
            all_gather_kernels.writer_kernel_index == kAllGatherKernels.writer_kernel_index,
        "all_gather_matmul_async: the all-gather reader/writer must be the first two kernels (got {}, {}); the "
        "cache-hit override depends on it",
        all_gather_kernels.reader_kernel_index,
        all_gather_kernels.writer_kernel_index);

    // The all-gather workers and the matmul cores are separate core sets (all_gather_core_grid_offset); the merge
    // checks that, and keeps the all-gather's kernels first.
    return tt::tt_metal::merge_program_descriptors({all_gather_desc, matmul_desc});
}

}  // namespace

tt::tt_metal::WorkloadDescriptor AllGatherMatmulAsyncProgramFactory::create_workload_descriptor(
    const AllGatherMatmulAsyncParams& operation_attributes,
    const AllGatherMatmulAsyncInputs& tensor_args,
    AllGatherMatmulAsyncResult& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    const auto coords = tensor_coords.coords();
    workload.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_program_descriptor(operation_attributes, coord, tensor_args, tensor_return_value)});
    }
    return workload;
}

AllGatherMatmulAsyncMeshWorkloadFactory::cached_mesh_workload_t
AllGatherMatmulAsyncMeshWorkloadFactory::create_mesh_workload(
    const AllGatherMatmulAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherMatmulAsyncInputs& tensor_args,
    AllGatherMatmulAsyncResult& tensor_return_value) {
    return descriptor_adapter_t::create_mesh_workload(
        operation_attributes, tensor_coords, tensor_args, tensor_return_value);
}

void AllGatherMatmulAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherMatmulAsyncParams& operation_attributes,
    const AllGatherMatmulAsyncInputs& tensor_args,
    AllGatherMatmulAsyncResult& tensor_return_value) {
    // Tensor addresses (all-gather input/output, matmul in0/in1/bias/output, tensor-backed CBs).
    descriptor_adapter_t::apply_descriptor(cached_workload, operation_attributes, tensor_args, tensor_return_value);

    // The caller-owned GlobalSemaphores.
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        ttnn::apply_all_gather_async_minimal_default_semaphore_args(
            program,
            kAllGatherKernels,
            operation_attributes.all_gather_async_attributes.barrier_semaphore,
            operation_attributes.all_gather_async_attributes.semaphore);
    }
}

}  // namespace ttnn::experimental::prim
