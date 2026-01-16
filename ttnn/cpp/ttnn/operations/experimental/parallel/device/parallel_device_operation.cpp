// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "parallel_device_operation.hpp"

namespace ttnn::operations::experimental::parallel {

// =============================================================================
// ParallelDeviceOperation Implementation
// =============================================================================

ParallelDeviceOperation::program_factory_t ParallelDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ParallelProgramFactory{};
}

void ParallelDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void ParallelDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    TT_FATAL(!operation_attributes.branches.empty(), "ParallelDeviceOperation requires at least one branch");

    // Validate that core ranges don't overlap between branches
    for (size_t i = 0; i < operation_attributes.branches.size(); ++i) {
        for (size_t j = i + 1; j < operation_attributes.branches.size(); ++j) {
            const auto& cores_i = operation_attributes.branches[i]->core_range;
            const auto& cores_j = operation_attributes.branches[j]->core_range;

            auto intersection = cores_i.intersection(cores_j);
            TT_FATAL(
                intersection.empty(),
                "Parallel branches must have disjoint core assignments. "
                "Branch {} and branch {} have overlapping cores.",
                i,
                j);
        }
    }

    // Validate each individual branch
    for (const auto& branch : operation_attributes.branches) {
        branch->check_on_cache_miss();
    }
}

spec_return_value_t ParallelDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    spec_return_value_t specs;
    specs.reserve(operation_attributes.branches.size());

    for (const auto& branch : operation_attributes.branches) {
        specs.push_back(branch->get_output_specs());
    }

    return specs;
}

tensor_return_value_t ParallelDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    tensor_return_value_t outputs;
    outputs.reserve(operation_attributes.branches.size());

    for (const auto& branch : operation_attributes.branches) {
        // Compute output specs, use for creating output tensors
        outputs.push_back(branch->make_output_tensors());
    }

    return outputs;
}

}  // namespace ttnn::operations::experimental::parallel

namespace ttnn::prim {

ttnn::operations::experimental::parallel::ParallelDeviceOperation::tensor_return_value_t parallel(
    const ttnn::operations::experimental::parallel::operation_attributes_t& operation_attributes) {
    using OperationType = ttnn::operations::experimental::parallel::ParallelDeviceOperation;
    auto tensor_args = OperationType::tensor_args_t{};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
