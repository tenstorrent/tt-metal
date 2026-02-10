// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed_mla_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/logger.hpp>
#include <tt-metalium/program.hpp>

namespace ttnn::operations::transformer::sdpa_prefill {

DistributedMLADeviceOperation::program_factory_t DistributedMLADeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return DistributedMLAProgram{};
}

void DistributedMLADeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    auto input_tensor = tensor_args.input_tensor;

    // Basic validations
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffer on device!");
    TT_FATAL(input_tensor.logical_shape().rank() >= 2, "DistributedMLA requires tensor of rank 2 or greater");
}

void DistributedMLADeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // For now, no additional validation on cache hit
}

DistributedMLADeviceOperation::spec_return_value_t DistributedMLADeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto output_shape = input_tensor.logical_shape();

    auto mem_config = operation_attributes.memory_config;
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), mem_config));
}

DistributedMLADeviceOperation::tensor_return_value_t DistributedMLADeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_specs, tensor_args.input_tensor.device());
}

DistributedMLADeviceOperation::DistributedMLAProgram::cached_mesh_workload_t
DistributedMLADeviceOperation::DistributedMLAProgram::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<DistributedMLADeviceOperation::DistributedMLAProgram::shared_variables_t>
DistributedMLADeviceOperation::DistributedMLAProgram::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program{};

    // Get device index using the CCL common function - this is the key part!
    log_debug(tt::LogOp, "Getting device index for coordinate {}", mesh_coordinate);
    uint32_t device_order = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, operation_attributes.cluster_axis);

    // Log the device order information
    log_info(
        tt::LogOp,
        "Device at coordinate {} has linearized index: {} (cluster_axis: {})",
        mesh_coordinate,
        device_order,
        operation_attributes.cluster_axis.has_value() ? std::to_string(operation_attributes.cluster_axis.value())
                                                      : "none");

    // For now, create a simple program that just copies the input to output
    // In the full implementation, this would contain the actual SDPA computation with Q offset based on device_order

    return {std::move(program), shared_variables_t{.device_order = device_order}};
}

void DistributedMLADeviceOperation::DistributedMLAProgram::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // For now, no runtime arguments to override
}

}  // namespace ttnn::operations::transformer::sdpa_prefill

namespace ttnn::prim {
ttnn::Tensor distributed_mla(
    const ttnn::Tensor& input_tensor, std::optional<uint32_t> cluster_axis, const ttnn::MemoryConfig& memory_config) {
    using OperationType = ttnn::operations::transformer::sdpa_prefill::DistributedMLADeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.cluster_axis = cluster_axis, .memory_config = memory_config},
        OperationType::tensor_args_t{.input_tensor = input_tensor});
}
}  // namespace ttnn::prim
