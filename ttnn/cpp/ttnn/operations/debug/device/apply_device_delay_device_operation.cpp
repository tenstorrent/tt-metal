// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "apply_device_delay_device_operation.hpp"

#include <cstddef>

#include <tt_stl/assert.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::debug {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

ApplyDeviceDelayDeviceOperation::program_factory_t ApplyDeviceDelayDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ApplyDeviceDelayMeshWorkload{};
}

void ApplyDeviceDelayDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    const auto& mesh_device = *operation_attributes.mesh_device;
    TT_FATAL(operation_attributes.mesh_device != nullptr, "mesh_device is nullptr");
    const auto& view = mesh_device.get_view();

    const auto& delays = operation_attributes.delays;

    TT_FATAL(view.is_mesh_2d(), "apply_device_delay currently supports only 2D mesh");
    TT_FATAL(
        delays.size() == view.num_rows(), "delays rows ({}) must match mesh rows ({})", delays.size(), view.num_rows());
    for (size_t r = 0; r < delays.size(); ++r) {
        TT_FATAL(
            delays[r].size() == view.num_cols(),
            "delays cols for row {} ({}) must match mesh cols ({})",
            r,
            delays[r].size(),
            view.num_cols());
    }
}

void ApplyDeviceDelayDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // No additional validation needed on cache hit
}

ApplyDeviceDelayDeviceOperation::spec_return_value_t ApplyDeviceDelayDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return std::vector<ttnn::TensorSpec>{};
}

ApplyDeviceDelayDeviceOperation::tensor_return_value_t ApplyDeviceDelayDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return std::vector<ttnn::Tensor>{};
}

ApplyDeviceDelayDeviceOperation::ApplyDeviceDelayMeshWorkload::cached_mesh_workload_t
ApplyDeviceDelayDeviceOperation::ApplyDeviceDelayMeshWorkload::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    log_info(tt::LogAlways, "Creating delay mesh workload");
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, cached_program.shared_variables);
    }
    log_info(tt::LogAlways, "Created delay mesh workload");
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<ApplyDeviceDelayDeviceOperation::ApplyDeviceDelayMeshWorkload::shared_variables_t>
ApplyDeviceDelayDeviceOperation::ApplyDeviceDelayMeshWorkload::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    log_info(tt::LogAlways, "Creating delay program at mesh coordinate: {}", mesh_coordinate);
    tt::tt_metal::Program program{};
    auto subdevice_cores = corerange_to_cores(operation_attributes.worker_core_range_set);
    auto kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/debug/device/kernels/dataflow/device_delay_spin.cpp",
        subdevice_cores.at(0),
        DataMovementConfig{.compile_args = {operation_attributes.delays[mesh_coordinate[0]][mesh_coordinate[1]]}});
    log_info(tt::LogAlways, "Created delay program at mesh coordinate: {}", mesh_coordinate);
    return {std::move(program), shared_variables_t{.kernel_id = kernel_id}};
}

void ApplyDeviceDelayDeviceOperation::ApplyDeviceDelayMeshWorkload::override_runtime_arguments(
    cached_mesh_workload_t& /*cached_workload*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // No runtime arguments to override for this operation since delay cycles are compile-time
}

}  // namespace ttnn::operations::debug

namespace ttnn::prim {

ttnn::operations::debug::ApplyDeviceDelayDeviceOperation::tensor_return_value_t apply_device_delay(
    ttnn::MeshDevice& mesh_device,
    const std::vector<std::vector<uint32_t>>& delays,
    const CoreRangeSet& subdevice_core_range_set) {
    using OperationType = ttnn::operations::debug::ApplyDeviceDelayDeviceOperation;

    log_info(tt::LogAlways, "Initializing delay op structs");
    auto operation_attributes = OperationType::operation_attributes_t{
        .delays = delays, .worker_core_range_set = subdevice_core_range_set, .mesh_device = &mesh_device};
    auto tensor_args = OperationType::tensor_args_t{};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
