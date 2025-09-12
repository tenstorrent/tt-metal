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

namespace ttnn::operations::ccl {

ReduceScatterDeviceOperation::ReduceScatterProgram::cached_mesh_workload_t
ReduceScatterDeviceOperation::ReduceScatterProgram::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        tt::tt_metal::Program program{};
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(program));
        shared_variables.emplace(coord, shared_variables_t{});
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<ReduceScatterDeviceOperation::ReduceScatterProgram::shared_variables_t>
ReduceScatterDeviceOperation::ReduceScatterProgram::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::Program program{};
    return {std::move(program), {}};
}

void ReduceScatterDeviceOperation::ReduceScatterProgram::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::ccl
