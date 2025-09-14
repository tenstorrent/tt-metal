// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"

namespace ttnn::operations::ccl {

ReduceScatterDeviceOperation::ReduceScatterProgram::cached_mesh_workload_t
ReduceScatterDeviceOperation::ReduceScatterProgram::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto mesh_device = tensor_args.input_tensor.device();
    auto sd_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    // create semaphores
    // 3 semaphores used for within op synchronizations
    std::vector<tt::tt_metal::GlobalSemaphore> multidevice_semaphores = {
        ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0),
        ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0),
        ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0),
    };
    // 1 barrier semaphore used to ensure that all the buffers are allocated
    ttnn::SmallVector<tt::tt_metal::SubDeviceId> subdevice_ids = {sd_id};
    auto barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0);
    tt::tt_metal::distributed::Synchronize(
        mesh_device, std::nullopt, subdevice_ids);  // interaction with subdevice needs to be investigated

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

ttnn::device_operation::CachedProgram<ReduceScatterDeviceOperation::ReduceScatterProgram::shared_variables_t>
ReduceScatterDeviceOperation::ReduceScatterProgram::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<tt::tt_metal::GlobalSemaphore>& multidevice_semaphores,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    tt::tt_metal::Program program{};

    // Get mesh and axis related information
    auto mesh_device = tensor_args.input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    const auto& mesh_shape = mesh_device->shape();
    auto boundary_mode = operation_attributes.topology == ttnn::ccl::Topology::Ring
                             ? tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP
                             : tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::NONE;
    uint32_t reduction_devices =
        operation_attributes.cluster_axis.value() == 0 ? mesh_view.num_rows() : mesh_view.num_cols();

    auto forward_coordinate =
        mesh_coordinate.get_neighbor(mesh_shape, 1, operation_attributes.cluster_axis.value(), boundary_mode);
    auto backward_coordinate =
        mesh_coordinate.get_neighbor(mesh_shape, -1, operation_attributes.cluster_axis.value(), boundary_mode);

    std::optional<IDevice*> forward_device =
        forward_coordinate.has_value() ? std::make_optional(mesh_device->get_device(forward_coordinate.value()))
                                       : std::nullopt;

    std::optional<IDevice*> backward_device =
        backward_coordinate.has_value() ? std::make_optional(mesh_device->get_device(backward_coordinate.value()))
                                        : std::nullopt;

    // Get core and subdevice related information
    auto sd_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    auto bbox = subdevice_core_range_set.bounding_box();
    auto first_coord = bbox.start_coord;

    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> no_fuse = std::nullopt;

    auto builder = operation_attributes.topology == ttnn::ccl::Topology::Ring
                       ? build_ring_reduce_scatter_minimal_async_program_artifacts
                       : build_line_reduce_scatter_minimal_async_program_artifacts;
    auto reduce_scatter_program_artifacts = builder(
        program,
        tensor_args.input_tensor,
        tensor_return_value.at(0),
        mesh_device->get_device(mesh_coordinate),
        forward_device,
        backward_device,
        tensor_return_value.at(0),
        operation_attributes.dim,
        operation_attributes.num_links,
        reduction_devices,
        mesh_coordinate[operation_attributes.cluster_axis.value()],
        operation_attributes.topology,
        multidevice_semaphores,
        barrier_semaphore,
        false,  // since we don't have a persistent intermediate buffer option, this must be false
        operation_attributes.subdevice_id,
        no_fuse,       // never fusing with this
        std::nullopt,  // use chunks per sync decision making tree
        std::nullopt,  // use num workers per link decision making tree
        std::nullopt,  // use num buffers per channel decision making tree
        first_coord);  // first core in the subdevice is our offset as we don't use this version for fusions

    return {
        std::move(program),
        {.multidevice_semaphores = multidevice_semaphores,
         .barrier_semaphore = barrier_semaphore,
         .program_artifacts = reduce_scatter_program_artifacts}};
}

void ReduceScatterDeviceOperation::ReduceScatterProgram::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::ccl
