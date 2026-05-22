// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include <tt_stl/overloaded.hpp>

namespace ttnn::experimental::prim {

namespace {

// Build the per-coord ProgramDescriptor for the fused AllGather + Matmul op.
// Mirrors the legacy create_at() body 1:1 but routes through the
// ProgramDescriptor variants of the matmul and AllGather builders.  The
// fused-op signaler plumbing between the two builders is identical to the
// legacy path: the matmul builder populates the signaler with its receiver
// cores/semaphores, which the AllGather builder then reads back.
tt::tt_metal::ProgramDescriptor build_descriptor_at(
    const Tensor& input_tensor,
    Tensor& all_gather_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,

    /* All Gather Params */
    const MeshCoordinate& target_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,

    /* Matmul Params */
    const std::optional<Tensor>& bias,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out) {
    tt::tt_metal::ProgramDescriptor desc;

    ////////////// Params for fused op signalers //////////////
    auto tensor_slicer =
        ttnn::ccl::InterleavedRingAllGatherTensorSlicer(input_tensor, all_gather_output_tensor, dim, ring_index);
    bool is_clockwise_direction = true;
    const uint32_t num_transfers = 4;
    const uint32_t weight_tensor_width = weight_tensor.padded_shape()[3] / 32;

    ////////////////////////////////////////////////////////

    // Create a matmul signal info object that gets populated by the matmul kernel
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(ttnn::experimental::ccl::MatmulFusedOpSignalerType::ALL_GATHER);
    matmul_fused_op_signaler->init_all_gather(
        num_transfers,
        ring_size,
        ring_index,
        tensor_slicer.num_cols,
        tensor_slicer.output_page_offset,
        is_clockwise_direction,
        tensor_slicer.num_cols *
            weight_tensor_width /* weight_output_page_offset: stride across a tensor slice in the weight_tensor */
    );

    std::visit(
        ttsl::overloaded{
            [&](const operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& config) {
                ttnn::prim::matmul_multi_core_reuse_mcast_2d_optimized_helper_descriptor(
                    desc,
                    all_gather_output_tensor,
                    weight_tensor,
                    bias,
                    matmul_output_tensor,
                    bcast_batch,
                    compute_kernel_config,
                    config,
                    untilize_out,
                    matmul_fused_op_signaler);
            },
            [&](const operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& config) {
                ttnn::prim::matmul_multi_core_reuse_mcast_1d_optimized_helper_descriptor(
                    desc,
                    all_gather_output_tensor,
                    {weight_tensor},
                    bias,
                    {matmul_output_tensor},
                    bcast_batch,
                    compute_kernel_config,
                    config,
                    untilize_out,
                    matmul_fused_op_signaler,
                    std::nullopt,
                    std::nullopt);
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
    (void)ttnn::build_all_gather_async_minimal_default_program_artifacts_descriptor(
        desc,
        input_tensor,
        target_device_coord,
        forward_coord,
        backward_coord,
        all_gather_output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        all_gather_fused_op_signaler,
        chunks_per_sync,
        num_workers_per_direction_opt,
        num_buffers_per_channel,
        core_grid_offset,
        false,  // reverse_order = false by default
        std::nullopt);

    return desc;
}

}  // namespace

// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
tt::tt_metal::WorkloadDescriptor AllGatherMatmulAsyncMeshWorkloadFactory::create_workload_descriptor(
    const AllGatherMatmulAsyncParams& operation_attributes,
    const AllGatherMatmulAsyncInputs& tensor_args,
    AllGatherMatmulAsyncResult& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    for (const auto& mesh_coord : tensor_coords.coords()) {
        const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};

        uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
            tensor_args.input_tensor, mesh_coord, operation_attributes.all_gather_async_attributes.cluster_axis);

        std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor,
            mesh_coord,
            1,
            operation_attributes.all_gather_async_attributes.topology,
            operation_attributes.all_gather_async_attributes.cluster_axis);

        std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor,
            mesh_coord,
            -1,
            operation_attributes.all_gather_async_attributes.topology,
            operation_attributes.all_gather_async_attributes.cluster_axis);

        auto desc = build_descriptor_at(
            tensor_args.input_tensor,
            tensor_return_value[0],
            tensor_args.weight_tensor,
            tensor_return_value[1],

            mesh_coord,
            forward_coord,
            backward_coord,
            operation_attributes.all_gather_async_attributes.dim,
            operation_attributes.all_gather_async_attributes.num_links,
            operation_attributes.all_gather_async_attributes.ring_size,
            device_index,
            operation_attributes.all_gather_async_attributes.topology,
            operation_attributes.all_gather_async_attributes.semaphore,
            operation_attributes.all_gather_async_attributes.barrier_semaphore,
            operation_attributes.all_gather_async_attributes.using_persistent_buffers,
            operation_attributes.all_gather_async_attributes.sub_device_id,
            operation_attributes.all_gather_async_attributes.chunks_per_sync,
            operation_attributes.all_gather_async_attributes.num_workers_per_link,
            operation_attributes.all_gather_async_attributes.num_buffers_per_channel,
            operation_attributes.all_gather_core_grid_offset,

            tensor_args.bias,
            operation_attributes.matmul.bcast_batch.value(),
            operation_attributes.matmul.compute_kernel_config.value(),
            operation_attributes.matmul.program_config.value(),
            operation_attributes.matmul.untilize_out);

        workload_descriptor.programs.push_back({single_coord_range, std::move(desc)});
    }

    return workload_descriptor;
}

}  // namespace ttnn::experimental::prim
