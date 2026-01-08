// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <unordered_map>
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"

namespace ttnn::operations::ccl {

AllGatherDeviceOperation::AllGatherProgram::cached_mesh_workload_t
AllGatherDeviceOperation::AllGatherProgram::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    // Create semaphores internally (internalized global semaphores)
    // 2 semaphores used for within op synchronizations (forward and backward links)
    std::vector<tt::tt_metal::GlobalSemaphore> multidevice_semaphores = {
        ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0),
        ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0),
    };

    // 1 barrier semaphore used to ensure that all the buffers are allocated
    ttnn::SmallVector<tt::tt_metal::SubDeviceId> subdevice_ids = {sd_id};
    auto barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevice_ids);

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            tensor_coords,
            multidevice_semaphores,
            barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<AllGatherDeviceOperation::AllGatherProgram::shared_variables_t>
AllGatherDeviceOperation::AllGatherProgram::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<tt::tt_metal::GlobalSemaphore>& multidevice_semaphores,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    tt::tt_metal::Program program{};

    // Get mesh and axis related information
    auto* mesh_device = tensor_args.input_tensor.device();
    uint32_t target_ring_size =
        ::ttnn::ccl::get_topological_dimension(tensor_args.input_tensor, operation_attributes.cluster_axis);

    log_debug(tt::LogOp, "Getting forward neighbor for {}", mesh_coordinate);
    const std::optional<MeshCoordinate> forward_coordinate = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1, operation_attributes.topology, operation_attributes.cluster_axis);

    log_debug(tt::LogOp, "Getting backward neighbor for {}", mesh_coordinate);
    const std::optional<MeshCoordinate> backward_coordinate = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor,
        mesh_coordinate,
        -1,
        operation_attributes.topology,
        operation_attributes.cluster_axis);
    TT_FATAL(
        forward_coordinate.has_value() || backward_coordinate.has_value(),
        "DEBUG: forward_coord or backward_coord is null");

    log_debug(tt::LogOp, "Getting device index for {}", mesh_coordinate);
    uint32_t device_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, operation_attributes.cluster_axis);
    log_debug(tt::LogOp, "Device index for {} is {}", mesh_coordinate, device_index);

    // Get core and subdevice related information
    auto sd_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    if (operation_attributes.sub_core_grid.has_value()) {
        subdevice_core_range_set = subdevice_core_range_set.intersection(operation_attributes.sub_core_grid.value());
    }
    auto bbox = subdevice_core_range_set.bounding_box();
    auto first_coord = bbox.start_coord;

    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> no_fuse = std::nullopt;

    // Build the program artifacts using the shared builder function
    auto all_gather_program_artifacts = build_all_gather_async_minimal_default_program_artifacts(
        program,
        tensor_args.input_tensor,
        mesh_coordinate,
        forward_coordinate,
        backward_coordinate,
        tensor_return_value,
        operation_attributes.dim,
        operation_attributes.num_links,
        target_ring_size,
        device_index,
        operation_attributes.topology,
        multidevice_semaphores,
        barrier_semaphore,
        false,  // using_persistent_buffers - false since we always barrier in this version
        operation_attributes.subdevice_id,
        no_fuse,  // never fusing with this
        operation_attributes.chunks_per_sync,
        operation_attributes.num_workers_per_link,
        operation_attributes.num_buffers_per_channel,
        first_coord,  // first core in the subdevice is our offset as we don't use this version for fusions
        false,        // reverse_order = false
        operation_attributes.sub_core_grid);

    return {
        std::move(program),
        {.multidevice_semaphores = multidevice_semaphores,
         .barrier_semaphore = barrier_semaphore,
         .program_artifacts = all_gather_program_artifacts}};
}

void AllGatherDeviceOperation::AllGatherProgram::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Update runtime arguments using the shared function
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& coord = range.start_coord();
        TT_FATAL(
            coord == range.end_coord(),
            "Expected single coordinate per program but got range of {} to {}",
            coord,
            range.end_coord());
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        all_gather_async_minimal_default_helper_override_runtime_arguments(
            program,
            shared_variables.program_artifacts.reader_kernel_id,
            shared_variables.program_artifacts.writer_kernel_id,
            shared_variables.program_artifacts.all_cores,
            operation_attributes.num_links,
            shared_variables.program_artifacts.num_directions_per_link,
            shared_variables.program_artifacts.num_workers_per_direction,
            shared_variables.program_artifacts.num_mux_cores_per_direction_per_link,
            shared_variables.program_artifacts.num_cores_per_link,
            shared_variables.barrier_semaphore,
            shared_variables.multidevice_semaphores,
            tensor_args.input_tensor,
            tensor_return_value);
    }
}

}  // namespace ttnn::operations::ccl
