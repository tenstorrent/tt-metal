// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_async_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/line_reduce_scatter_minimal_async_program.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::constants;

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async::program {

MatmulReduceScatterAsyncProgramCallbacks matmul_reduce_scatter_async_multi_core_with_workers(
    /* General Params */
    const Tensor& input_tensor,
    Tensor& persistent_intermediate_tensor,
    Tensor& reduce_scatter_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,
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
    const CoreCoord core_grid_offset,

    /* Matmul Params */
    const std::optional<const Tensor>& bias,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out) {
    tt::tt_metal::Program program{};

    // Create the reduce scatter fused op signaler
    auto reduce_scatter_fused_op_signaler = ttnn::experimental::ccl::ReduceScatterFusedOpSignaler();
    reduce_scatter_fused_op_signaler.init_fused_op();

    // Reduce Scatter
    tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_program_with_callbacks =
        ttnn::create_line_reduce_scatter_minimal_async_program(
            program,
            matmul_output_tensor,
            persistent_intermediate_tensor,
            target_device_coord,
            forward_coord,
            backward_coord,
            reduce_scatter_output_tensor,
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
            core_grid_offset);
    const auto reduce_scatter_override_runtime_arguments_callback =
        reduce_scatter_program_with_callbacks.override_runtime_arguments_callback;

    // Create a matmul signal info object that gets populated by the matmul kernel
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::REDUCE_SCATTER);
    matmul_fused_op_signaler->init_reduce_scatter(
        reduce_scatter_fused_op_signaler->fused_op_receiver_cores_noc,
        reduce_scatter_fused_op_signaler->fused_op_receiver_signal_semaphores,
        reduce_scatter_fused_op_signaler->fused_op_signaler_mode);

    // Matmul
    tt::tt_metal::operation::ProgramWithCallbacks matmul_program_with_callbacks =
        operations::matmul::matmul_multi_core_reuse_mcast_2d_optimized_helper(
            reduce_scatter_program_with_callbacks.program,
            input_tensor,
            weight_tensor,
            bias,
            matmul_output_tensor,
            bcast_batch,
            compute_kernel_config,
            program_config,
            untilize_out,
            matmul_fused_op_signaler);
    const auto matmul_override_runtime_arguments_callback =
        matmul_program_with_callbacks.override_runtime_arguments_callback;

    // Return the program and individual callbacks separately
    return MatmulReduceScatterAsyncProgramCallbacks{
        .program = std::move(matmul_program_with_callbacks.program),
        .matmul_override_callback = matmul_override_runtime_arguments_callback,
        .reduce_scatter_override_callback = reduce_scatter_override_runtime_arguments_callback};
}

MatmulReduceScatterAsyncMeshWorkloadFactory::cached_mesh_workload_t
MatmulReduceScatterAsyncMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange single_mesh_coord_range{mesh_coord, mesh_coord};

            std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
                tensor_args.input_tensor,
                mesh_coord,
                1,
                operation_attributes.reduce_scatter_minimal_async_struct.topology,
                operation_attributes.reduce_scatter_minimal_async_struct.cluster_axis);

            std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
                tensor_args.input_tensor,
                mesh_coord,
                -1,
                operation_attributes.reduce_scatter_minimal_async_struct.topology,
                operation_attributes.reduce_scatter_minimal_async_struct.cluster_axis);

            uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
                tensor_args.input_tensor,
                mesh_coord,
                operation_attributes.reduce_scatter_minimal_async_struct.cluster_axis);

            auto& matmul_output = std::get<0>(tensor_return_value);
            auto& reduce_scatter_output = std::get<1>(tensor_return_value);

            auto program_callbacks = ttnn::operations::experimental::ccl::matmul_reduce_scatter_async::program::
                matmul_reduce_scatter_async_multi_core_with_workers(
                    tensor_args.input_tensor,
                    tensor_args.persistent_intermediate_buffer,
                    reduce_scatter_output,
                    tensor_args.weight_tensor,
                    matmul_output,
                    mesh_coord,
                    forward_coord,
                    backward_coord,
                    operation_attributes.reduce_scatter_minimal_async_struct.dim,
                    operation_attributes.reduce_scatter_minimal_async_struct.num_links,
                    operation_attributes.reduce_scatter_minimal_async_struct.ring_size,
                    device_index,
                    operation_attributes.reduce_scatter_minimal_async_struct.topology,
                    operation_attributes.reduce_scatter_minimal_async_struct.semaphore,
                    operation_attributes.reduce_scatter_minimal_async_struct.barrier_semaphore,
                    operation_attributes.reduce_scatter_minimal_async_struct.using_persistent_buffers,
                    operation_attributes.reduce_scatter_minimal_async_struct.sub_device_id,
                    operation_attributes.reduce_scatter_core_grid_offset,
                    tensor_args.bias,
                    operation_attributes.matmul_struct.bcast_batch.value(),
                    operation_attributes.matmul_struct.compute_kernel_config.value(),
                    operation_attributes.matmul_struct.program_config.value(),
                    operation_attributes.matmul_struct.untilize_out);

            shared_variables_t shared_vars{
                .matmul_override_callback = program_callbacks.matmul_override_callback,
                .reduce_scatter_override_callback = program_callbacks.reduce_scatter_override_callback,
                .mesh_coord = mesh_coord};
            shared_variables[single_mesh_coord_range] = std::move(shared_vars);
            workload.add_program(single_mesh_coord_range, std::move(program_callbacks.program));
        }
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void MatmulReduceScatterAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& matmul_output = std::get<0>(tensor_return_value);
    auto& reduce_scatter_output = std::get<1>(tensor_return_value);

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        // Get non-const reference to shared_vars to allow updates
        auto& shared_vars = cached_workload.shared_variables.at(range);

        // Call matmul override callback with stored shared parameters
        if (shared_vars.matmul_override_callback.has_value()) {
            std::vector<Tensor> matmul_input_tensors = {tensor_args.input_tensor, tensor_args.weight_tensor};
            std::vector<std::optional<const Tensor>> matmul_optional_input_tensors;
            if (tensor_args.bias.has_value()) {
                matmul_optional_input_tensors.push_back(tensor_args.bias.value());
            } else {
                matmul_optional_input_tensors.push_back(std::nullopt);
            }
            std::vector<Tensor> matmul_output_tensors = {matmul_output};

            shared_vars.matmul_override_callback.value()(
                nullptr, program, matmul_input_tensors, matmul_optional_input_tensors, matmul_output_tensors);
        }

        // Call reduce_scatter override callback with stored shared parameters
        if (shared_vars.reduce_scatter_override_callback.has_value()) {
            std::vector<Tensor> reduce_scatter_input_tensors = {matmul_output};
            std::vector<std::optional<const Tensor>> reduce_scatter_optional_input_tensors = {};
            std::vector<Tensor> reduce_scatter_output_tensors = {
                tensor_args.persistent_intermediate_buffer, reduce_scatter_output};

            shared_vars.reduce_scatter_override_callback.value()(
                nullptr,
                program,
                reduce_scatter_input_tensors,
                reduce_scatter_optional_input_tensors,
                reduce_scatter_output_tensors);
        }
    }
}

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async::program
