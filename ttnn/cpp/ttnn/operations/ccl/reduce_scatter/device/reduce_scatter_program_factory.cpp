// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_line_program_factory.hpp"

namespace ttnn::operations::ccl {

tt::tt_metal::WorkloadDescriptor ReduceScatterDeviceOperation::ReduceScatterProgram::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    // create semaphores
    // 3 semaphores used for within op synchronizations
    auto sem_buffer_type = operation_attributes.use_l1_small_for_semaphores ? tt::tt_metal::BufferType::L1_SMALL
                                                                            : tt::tt_metal::BufferType::L1;
    workload_descriptor.semaphores = {
        ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0, sem_buffer_type),
        ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0, sem_buffer_type),
        ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0, sem_buffer_type),
    };
    // 1 barrier semaphore used to ensure that all the buffers are allocated
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevice_ids = {sd_id};
    workload_descriptor.semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0, sem_buffer_type));
    const auto& multidevice_semaphores = workload_descriptor.semaphores;
    const auto& barrier_semaphore = workload_descriptor.semaphores.back();
    tt::tt_metal::distributed::Synchronize(
        *mesh_device, std::nullopt, subdevice_ids);  // interaction with subdevice needs to be investigated

    workload_descriptor.programs.reserve(tensor_coords.coords().size());
    for (const auto& mesh_coordinate : tensor_coords.coords()) {
        tt::tt_metal::ProgramDescriptor program{};

        // Get mesh and axis related information
        uint32_t target_ring_size =
            ::ttnn::ccl::get_topological_dimension(tensor_args.input_tensor, operation_attributes.cluster_axis);

        log_debug(tt::LogOp, "Getting forward neighbor for {}", mesh_coordinate);
        const std::optional<MeshCoordinate> forward_coordinate = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor,
            mesh_coordinate,
            1,
            operation_attributes.topology,
            operation_attributes.cluster_axis);

        log_debug(tt::LogOp, "Getting backward neighbor for {}", mesh_coordinate);
        const std::optional<MeshCoordinate> backward_coordinate =
            ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
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
        auto bbox = subdevice_core_range_set.bounding_box();
        auto first_coord = bbox.start_coord;

        std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> no_fuse = std::nullopt;

        // semaphores[0..2] are the op semaphores; semaphores[3] is the barrier. Pass only the first three.
        const std::vector<tt::tt_metal::GlobalSemaphore> op_semaphores(
            multidevice_semaphores.begin(), multidevice_semaphores.begin() + 3);
        if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
            ttnn::experimental::prim::build_ring_reduce_scatter_minimal_async_program_artifacts(
                program,
                tensor_args.input_tensor,
                tensor_return_value.at(0),
                /*penult_intermediate_tensor=*/std::nullopt,  // accessible via the reduce_scatter_minimal_async path
                mesh_coordinate,
                forward_coordinate,
                backward_coordinate,
                tensor_return_value.at(1),
                operation_attributes.dim,
                operation_attributes.num_links,
                target_ring_size,
                device_index,
                operation_attributes.topology,
                op_semaphores,
                barrier_semaphore,
                false,  // since we don't have a persistent intermediate buffer option, this must be false
                operation_attributes.subdevice_id,
                no_fuse,  // never fusing with this
                operation_attributes.chunks_per_sync,
                operation_attributes.num_workers_per_link,
                operation_attributes.num_buffers_per_channel,
                first_coord,  // first core in the subdevice is our offset as we don't use this version for fusions
                operation_attributes.compute_kernel_config);
        } else {
            ttnn::experimental::prim::build_line_reduce_scatter_minimal_async_program_artifacts(
                program,
                tensor_args.input_tensor,
                tensor_return_value.at(0),
                /*penult_intermediate_tensor=*/std::nullopt,  // accessible via the reduce_scatter_minimal_async path
                mesh_coordinate,
                forward_coordinate,
                backward_coordinate,
                tensor_return_value.at(1),
                operation_attributes.dim,
                operation_attributes.num_links,
                target_ring_size,
                device_index,
                operation_attributes.topology,
                op_semaphores,
                barrier_semaphore,
                false,  // since we don't have a persistent intermediate buffer option, this must be false
                operation_attributes.subdevice_id,
                no_fuse,  // never fusing with this
                operation_attributes.chunks_per_sync,
                operation_attributes.num_workers_per_link,
                operation_attributes.num_buffers_per_channel,
                first_coord,  // first core in the subdevice is our offset as we don't use this version for fusions
                operation_attributes.compute_kernel_config);
        }
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(mesh_coordinate), std::move(program)});
    }

    return workload_descriptor;
}

}  // namespace ttnn::operations::ccl
