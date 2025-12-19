// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_default_program_factory.hpp"

#include "ttnn/operations/experimental/ccl/composite_common.hpp"

namespace ttnn {

using namespace ccl;

namespace operations::experimental::ccl::all_gather_async {

DefaultMeshWorkloadFactory::cached_mesh_workload_t DefaultMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, output_tensor);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

DefaultMeshWorkloadFactory::cached_program_t DefaultMeshWorkloadFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;

    const auto& sender_device_coord = mesh_coordinate;  // coord
    const auto& forward_coord = get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    const auto& backward_coord = get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    const auto& dim = operation_attributes.dim;
    const auto& num_links = operation_attributes.num_links;
    const auto& ring_size = operation_attributes.ring_size;
    const auto& ring_index = get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);  // device_index
    const auto& topology = operation_attributes.topology;
    const auto& semaphore = operation_attributes.semaphore;
    const auto& barrier_semaphore = operation_attributes.barrier_semaphore;
    bool using_persistent_buffers = operation_attributes.using_persistent_buffers;
    const auto& sub_device_id = operation_attributes.sub_device_id;
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler;
    const auto& chunks_per_sync = operation_attributes.chunks_per_sync;
    const auto& num_workers_per_direction_opt = operation_attributes.num_workers_per_link;
    const auto& num_buffers_per_channel = operation_attributes.num_buffers_per_channel;
    const auto& core_grid_offset = CoreCoord(0, 0);
    const auto& reverse_order = operation_attributes.reverse_order;
    const auto& sub_core_grid = operation_attributes.sub_core_grid;

    log_trace(tt::LogOp, "Detected all gather specialized shape. all_gather_async_minimal_default is called");

    tt::tt_metal::Program program{};

    auto
        [reader_kernel_id,
         writer_kernel_id,
         all_cores,
         num_directions_per_link,
         num_workers_per_direction,
         num_mux_cores_per_direction_per_link,
         num_cores_per_link] =
            build_all_gather_async_minimal_default_program_artifacts(
                program,
                input_tensor,
                sender_device_coord,
                forward_coord,
                backward_coord,
                output_tensor,
                dim,
                num_links,
                ring_size,
                ring_index,
                topology,
                semaphore,
                barrier_semaphore,
                using_persistent_buffers,
                sub_device_id,
                fused_op_signaler,
                chunks_per_sync,
                num_workers_per_direction_opt,
                num_buffers_per_channel,
                core_grid_offset,
                reverse_order,
                sub_core_grid);

    return {
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .all_cores = all_cores,
            .num_directions_per_link = num_directions_per_link,
            .num_workers_per_direction = num_workers_per_direction,
            .num_mux_cores_per_direction_per_link = num_mux_cores_per_direction_per_link,
            .num_cores_per_link = num_cores_per_link}};
}

void DefaultMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    // Update runtime arguments for each program in the mesh workload
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        const auto& input = tensor_args.input_tensor;
        const auto& output = output_tensor;

        auto semaphore = operation_attributes.semaphore;
        auto barrier_semaphore = operation_attributes.barrier_semaphore;

        all_gather_async_minimal_default_helper_override_runtime_arguments(
            program,
            shared_vars.reader_kernel_id,
            shared_vars.writer_kernel_id,
            shared_vars.all_cores,
            operation_attributes.num_links,
            shared_vars.num_directions_per_link,
            shared_vars.num_workers_per_direction,
            shared_vars.num_mux_cores_per_direction_per_link,
            shared_vars.num_cores_per_link,
            barrier_semaphore,
            semaphore,
            input,
            output);
    }
}

}  // namespace operations::experimental::ccl::all_gather_async

}  // namespace ttnn
