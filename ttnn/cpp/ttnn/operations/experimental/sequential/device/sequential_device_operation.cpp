// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sequential_device_operation.hpp"

namespace ttnn::operations::experimental::sequential {

// =============================================================================
// SequentialDeviceOperation Implementation
// =============================================================================

SequentialDeviceOperation::program_factory_t SequentialDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return SequentialProgramFactory{};
}

void SequentialDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void SequentialDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    TT_FATAL(!operation_attributes.steps.empty(), "SequentialDeviceOperation requires at least one step");

    // Validate each individual step
    for (const auto& step : operation_attributes.steps) {
        TT_FATAL(!step->get_cores().empty(), "Step {} requires a non-empty core range", step->operation_name());
        step->check_on_cache_miss();
    }
}

spec_return_value_t SequentialDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // Sequential returns outputs from the LAST step only
    if (operation_attributes.steps.empty()) {
        return {};
    }

    return operation_attributes.steps.back()->get_output_specs();
}

tensor_return_value_t SequentialDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // Sequential returns outputs from the LAST step only
    if (operation_attributes.steps.empty()) {
        return {};
    }

    return operation_attributes.steps.back()->make_output_tensors();
}

ttsl::hash::hash_t SequentialDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // Hash together the operation type, number of steps, and each step's core range
    ttsl::hash::hash_t combined_hash = typeid(SequentialDeviceOperation).hash_code();

    // Hash each step (type and cores)
    for (const auto& step : operation_attributes.steps) {
        combined_hash = ttsl::hash::hash_objects(combined_hash, step->type_info().hash_code());

        // Hash the step's core range
        for (const auto& range : step->get_cores().ranges()) {
            combined_hash = ttsl::hash::hash_objects(
                combined_hash, range.start_coord.x, range.start_coord.y, range.end_coord.x, range.end_coord.y);
        }
    }

    return combined_hash;
}

}  // namespace ttnn::operations::experimental::sequential

namespace ttnn::prim {

ttnn::operations::experimental::sequential::SequentialDeviceOperation::tensor_return_value_t sequential(
    const ttnn::operations::experimental::sequential::operation_attributes_t& operation_attributes) {
    using OperationType = ttnn::operations::experimental::sequential::SequentialDeviceOperation;
    auto tensor_args = OperationType::tensor_args_t{};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
