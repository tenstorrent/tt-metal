// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"  //TODO: migrate this code to use new matmul API. This code relies on the old matmul struct
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/core_coord.hpp>
#include <unordered_map>

namespace ttnn::operations::experimental::ccl::all_gather_matmul_async::program {

using Tensors = std::vector<Tensor>;

// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
AllGatherMatmulAsyncMeshWorkloadFactory::cached_program_t AllGatherMatmulAsyncMeshWorkloadFactory::create_at(
    const Tensor& input_tensor,
    Tensor& all_gather_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,

    /* All Gather Params */
    IDevice* target_device,
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
    bool untilize_out

) {
    tt::tt_metal::Program program{};

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

    // Matmul
    std::optional<tt::tt_metal::operation::ProgramWithCallbacks>
        matmul_program_with_callbacks;  // TODO: migrate to not use the old program_with_callbacks in matmul old program
                                        // factories
    std::optional<tt::tt_metal::operation::OverrideRuntimeArgumentsCallback<Tensors>>
        matmul_override_runtime_arguments_callback;  // TODO: migrate to not use the old
                                                     // override_runtime_arguments_callback in matmul old program
                                                     // factories

    std::visit(
        [&](const auto& config) {
            using ProgramConfigType = std::decay_t<decltype(config)>;
            if (std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                matmul_program_with_callbacks = operations::matmul::matmul_multi_core_reuse_mcast_2d_optimized_helper(
                    program,
                    all_gather_output_tensor,
                    weight_tensor,
                    bias,
                    matmul_output_tensor,
                    bcast_batch,
                    compute_kernel_config,
                    config,
                    untilize_out,
                    matmul_fused_op_signaler);  // TODO: migrate to not use the old program_with_callbacks in matmul old
                                                // program factories
                matmul_override_runtime_arguments_callback =
                    matmul_program_with_callbacks
                        ->override_runtime_arguments_callback;  // TODO: migrate to not use the old
                                                                // override_runtime_arguments_callback in matmul old
                                                                // program factories
            } else if (std::is_same_v<
                           ProgramConfigType,
                           operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                matmul_program_with_callbacks = operations::matmul::matmul_multi_core_reuse_mcast_1d_optimized_helper(
                    program,
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
                    std::nullopt);  // TODO: migrate to not use the old program_with_callbacks in matmul old program
                                    // factories
                matmul_override_runtime_arguments_callback =
                    matmul_program_with_callbacks
                        ->override_runtime_arguments_callback;  // TODO: migrate to not use the old
                                                                // override_runtime_arguments_callback in matmul old
                                                                // program factories
            } else {
                TT_THROW("Unsupported MatmulProgramConfig type. Needs to be 1D or 2D Multicast.");
            }
        },
        program_config);

    if (!matmul_program_with_callbacks.has_value()) {
        TT_THROW("Matmul program with callbacks not created");  // TODO: migrate to not use the old
                                                                // matmul_program_with_callbacks in matmul old program
                                                                // factories
    }

    // Create the all gather fused op signaler
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::AllGatherFusedOpSignaler();
    all_gather_fused_op_signaler->init_fused_op(
        matmul_fused_op_signaler->fused_op_receiver_cores_noc,
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores,
        matmul_fused_op_signaler->fused_op_signaler_mode);

    // All Gather
    auto all_gather_async_shared_variables = ttnn::build_all_gather_async_minimal_default_program_artifacts(
        matmul_program_with_callbacks->program,
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

    return cached_program_t(
        {std::move(matmul_program_with_callbacks->program),
         shared_variables_t{
             .all_gather_async_shared_variables = all_gather_async_shared_variables,
             // TODO: migrate to not use the old
             // override_runtime_arguments_callback in matmul program factories
             .matmul_override_runtime_arguments_callback = matmul_override_runtime_arguments_callback}});
}

AllGatherMatmulAsyncMeshWorkloadFactory::cached_mesh_workload_t
AllGatherMatmulAsyncMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& mesh_coord : tensor_coords.coords()) {
        const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
        auto* mesh_device = tensor_args.input_tensor.device();
        IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coord) : tensor_args.input_tensor.device();

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

        auto cached_program = create_at(
            tensor_args.input_tensor,
            tensor_return_value[0],
            tensor_args.weight_tensor,
            tensor_return_value[1],

            target_device,
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

        workload.add_program(single_coord_range, std::move(cached_program.program));
        shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void AllGatherMatmulAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Fuse the override runtime arguments callbacks
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        if (shared_vars.matmul_override_runtime_arguments_callback.has_value()) {
            shared_vars.matmul_override_runtime_arguments_callback
                .value()(  // TODO: migrate to not use the old override_runtime_arguments_callback in matmul old program
                           // factories
                    &operation_attributes,
                    program,
                    {tensor_return_value[0], tensor_args.weight_tensor}, /* all gather output tensor, weight tensor */
                    {tensor_args.bias},
                    {tensor_return_value[1]} /* matmul output tensor */
                );
        }

        auto& all_gather_async_shared_variables = shared_vars.all_gather_async_shared_variables;
        const auto& all_gather_async_attributes = operation_attributes.all_gather_async_attributes;
        all_gather_async_minimal_default_helper_override_runtime_arguments(
            program,
            all_gather_async_shared_variables.reader_kernel_id,
            all_gather_async_shared_variables.writer_kernel_id,
            all_gather_async_shared_variables.all_cores,
            all_gather_async_attributes.num_links,
            all_gather_async_shared_variables.num_directions_per_link,
            all_gather_async_shared_variables.num_workers_per_direction,
            all_gather_async_shared_variables.num_mux_cores_per_direction_per_link,
            all_gather_async_shared_variables.num_cores_per_link,
            all_gather_async_attributes.barrier_semaphore,
            all_gather_async_attributes.semaphore,
            tensor_args.input_tensor,
            tensor_return_value[0]);
    }
}

}  // namespace ttnn::operations::experimental::ccl::all_gather_matmul_async::program
