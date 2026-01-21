// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/device_operation.hpp"
#include "sequential_device_operation_types.hpp"
#include "sequential_factory.hpp"

namespace ttnn::experimental::prim {

// =============================================================================
// SequentialDeviceOperation
// =============================================================================

struct SequentialDeviceOperation {
    using operation_attributes_t = SequentialParams;
    // tensor_args_t is empty since actual tensors are in StepDescriptors
    using tensor_args_t = SequentialInputs;
    // Sequential returns the outputs from the LAST step only
    using tensor_return_value_t = std::vector<Tensor>;
    // Output specs from the last step
    using spec_return_value_t = std::vector<TensorSpec>;
    using program_factory_t = std::variant<SequentialProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    // Hash together all step hashes
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Launch the sequential device operation
std::vector<Tensor> sequential(const ttnn::experimental::prim::SequentialParams& operation_attributes);

}  // namespace ttnn::prim
