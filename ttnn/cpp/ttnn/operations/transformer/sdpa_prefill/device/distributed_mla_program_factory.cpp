// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed_mla_program_factory.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>

namespace ttnn::operations::transformer::sdpa_prefill {

DistributedMlaMeshWorkloadFactory::cached_mesh_workload_t DistributedMlaMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // auto* mesh_device = tensor_args.input_tensor.device();

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

DistributedMlaMeshWorkloadFactory::cached_program_t DistributedMlaMeshWorkloadFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    tt::tt_metal::Program program{};

    // Get device order using physical coordinate
    uint32_t device_order = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, operation_attributes.cluster_axis);

    // Log device order for verification
    log_info(
        tt::LogOp,
        "Device at coordinate MeshCoordinate([{}, {}]) has linearized index: {} (cluster_axis: {})",
        mesh_coordinate[0],
        mesh_coordinate[1],
        device_order,
        operation_attributes.cluster_axis.value_or(0));

    // In the full implementation, this would contain the actual SDPA computation with Q offset based on device_order

    return {std::move(program), shared_variables_t{.device_order = device_order}};
}

void DistributedMlaMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& /*cached_workload*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // For now, no runtime arguments to override
}

}  // namespace ttnn::operations::transformer::sdpa_prefill
