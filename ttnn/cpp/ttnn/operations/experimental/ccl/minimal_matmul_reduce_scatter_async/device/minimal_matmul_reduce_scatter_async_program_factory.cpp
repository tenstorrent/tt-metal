// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_reduce_scatter_async_program_factory.hpp"

#include <algorithm>

#include <optional>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_line_program_factory.hpp"

namespace ttnn::operations::experimental::ccl::minimal_matmul_reduce_scatter_async::program {

MinimalMatmulReduceScatterAsyncProgramFactory::cached_mesh_workload_t
MinimalMatmulReduceScatterAsyncProgramFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_vars;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(args, coord, tensor_args, output_tensors);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_vars.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_vars)};
}

MinimalMatmulReduceScatterAsyncProgramFactory::cached_program_t
MinimalMatmulReduceScatterAsyncProgramFactory::create_at(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinate& mesh_coord,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    ttnn::ccl::Topology topology = args.reduce_scatter_params.topology;

    const auto& dim = args.reduce_scatter_params.dim;
    const auto& num_links = args.reduce_scatter_params.num_links;
    const auto& ring_size = args.reduce_scatter_params.ring_size;
    const auto& semaphore = args.reduce_scatter_params.semaphore;
    const auto& barrier_semaphore = args.reduce_scatter_params.barrier_semaphore;
    const auto& using_persistent_buffers = args.reduce_scatter_params.using_persistent_buffers;
    const auto& sub_device_id = args.reduce_scatter_params.sub_device_id;

    auto compute_kernel_config = args.matmul_struct.compute_kernel_config;

    tt::tt_metal::Program program{};

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
    reduce_scatter_fused_op_signaler->is_minimal_matmul = true;
    reduce_scatter_fused_op_signaler->grid_size = args.matmul_struct.config->compute_with_storage_grid_size;
    reduce_scatter_fused_op_signaler->M_block_size = args.matmul_struct.config->M_block_size;
    // Reduce Scatter - use the new artifacts-based helper
    auto reduce_scatter_artifacts = ttnn::experimental::prim::build_ring_reduce_scatter_minimal_async_program_artifacts(
        program,
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
        args.reduce_scatter_core_grid_offset);

    // Create a matmul signal info object that gets populated by the matmul kernel
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::REDUCE_SCATTER);

    matmul_fused_op_signaler->init_reduce_scatter(
        reduce_scatter_fused_op_signaler->fused_op_receiver_cores_noc,
        reduce_scatter_fused_op_signaler->fused_op_receiver_signal_semaphores,
        reduce_scatter_fused_op_signaler->fused_op_signaler_mode);

    // Matmul
    auto matmul_shared_variables = ttnn::experimental::prim::minimal_matmul_factory_helper(
        program,
        tensor_args.input,
        tensor_args.weight,
        tensor_args.bias,
        args.matmul_struct.fused_activation,
        args.matmul_struct.config,
        output_tensors.mm,
        compute_kernel_config,
        matmul_fused_op_signaler);

    return cached_program_t{
        std::move(program),
        {.reduce_scatter_artifacts = std::move(reduce_scatter_artifacts),
         .matmul_shared_variables = std::move(matmul_shared_variables)}};
}

void MinimalMatmulReduceScatterAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        ttnn::experimental::prim::MinimalMatmulInputs matmul_inputs{
            .input_tensor = tensor_args.input, .weight_tensor = tensor_args.weight, .bias_tensor = {tensor_args.bias},
            // .optional_output_tensors = {output_tensors.mm}
        };
        ttnn::experimental::prim::MinimalMatmulProgramFactory::override_runtime_arguments(
            program, shared_vars.matmul_shared_variables, matmul_inputs, output_tensors.mm);

        // Call reduce scatter runtime arguments override directly using artifacts
        ttnn::experimental::prim::ring_reduce_scatter_minimal_async_helper_override_runtime_arguments(
            program,
            shared_vars.reduce_scatter_artifacts.reader_kernel_id,
            shared_vars.reduce_scatter_artifacts.writer_kernel_id,
            shared_vars.reduce_scatter_artifacts.all_cores,
            args.reduce_scatter_params.num_links,
            shared_vars.reduce_scatter_artifacts.num_directions_per_link,
            shared_vars.reduce_scatter_artifacts.num_workers_per_direction,
            shared_vars.reduce_scatter_artifacts.num_mux_cores_per_direction_per_link,
            shared_vars.reduce_scatter_artifacts.num_cores_per_link,
            args.reduce_scatter_params.barrier_semaphore,
            args.reduce_scatter_params.semaphore,
            output_tensors.mm,
            tensor_args.persistent_intermediate,
            output_tensors.reduce_scatter);
    }
}

}  // namespace ttnn::operations::experimental::ccl::minimal_matmul_reduce_scatter_async::program
