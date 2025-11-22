// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/ring_reduce_scatter_minimal_async_program.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/line_reduce_scatter_minimal_async_program.hpp"

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

    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> no_fuse = std::nullopt;

    // Convert operation_attributes to reduce_scatter_minimal_async format
    // Note: Can't directly reuse due to different struct types with different fields
    ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t minimal_async_attrs{
        .dim = operation_attributes.dim,
        .num_links = operation_attributes.num_links,
        .ring_size = target_ring_size,
        .output_mem_config = operation_attributes.memory_config,
        .optional_intermediate_mem_config = std::nullopt,
        .topology = operation_attributes.topology,
        .semaphore = multidevice_semaphores,
        .barrier_semaphore = barrier_semaphore,
        .using_persistent_buffers = false,  // since we don't have a persistent intermediate buffer option
        .sub_device_id = operation_attributes.subdevice_id,
        .cluster_axis = operation_attributes.cluster_axis,
        .chunks_per_sync = std::nullopt,         // use decision making tree
        .num_workers_per_link = std::nullopt,    // use decision making tree
        .num_buffers_per_channel = std::nullopt  // use decision making tree
    };

    ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::tensor_args_t minimal_async_tensor_args{
        .input_tensor = tensor_args.input_tensor, .persistent_output_buffers = std::nullopt};

    if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
        // Ring topology - use ring factory
        using RingFactory = ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::ring::
            RingReduceScatterMinimalAsyncProgramFactory;
        auto cached_program = RingFactory::create_at(
            minimal_async_attrs, mesh_coordinate, minimal_async_tensor_args, tensor_return_value);
        return {
            std::move(cached_program.program),
            {.multidevice_semaphores = multidevice_semaphores,
             .barrier_semaphore = barrier_semaphore,
             .program_artifacts = cached_program.shared_variables}};
    } else {
        // Line topology - use line factory
        using LineFactory = ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::line::
            LineReduceScatterMinimalAsyncProgramFactory;
        auto cached_program = LineFactory::create_at(
            minimal_async_attrs, mesh_coordinate, minimal_async_tensor_args, tensor_return_value);
        return {
            std::move(cached_program.program),
            {.multidevice_semaphores = multidevice_semaphores,
             .barrier_semaphore = barrier_semaphore,
             .program_artifacts = cached_program.shared_variables}};
    }
}

void ReduceScatterDeviceOperation::ReduceScatterProgram::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program;
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& coord = range.start_coord();
        TT_FATAL(
            coord == range.end_coord(),
            "Expected single coordinate per program but got range of {} to {}",
            coord,
            range.end_coord());
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
            using namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::ring;
            override_runtime_args(
                program,
                shared_variables.program_artifacts,
                shared_variables.barrier_semaphore,
                shared_variables.multidevice_semaphores,
                tensor_args.input_tensor,
                tensor_return_value.at(0),
                tensor_return_value.at(1));
        } else {
            line::override_runtime_args(
                program,
                shared_variables.program_artifacts,
                shared_variables.barrier_semaphore,
                shared_variables.multidevice_semaphores,
                tensor_args.input_tensor,
                tensor_return_value.at(0),
                tensor_return_value.at(1));
        }
    }
}

}  // namespace ttnn::operations::ccl
