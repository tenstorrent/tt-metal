// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/device_operation.hpp"
#include "parallel_device_operation_types.hpp"
#include "parallel_factory.hpp"

namespace ttnn::experimental::prim {

// =============================================================================
// ParallelDeviceOperation
// =============================================================================

struct ParallelDeviceOperation {
    using operation_attributes_t = ParallelParams;
    // tensor_args_t is empty since actual tensors are in BranchDescriptors
    using tensor_args_t = ParallelInputs;
    // Each inner vector is the outputs from one branch
    using tensor_return_value_t = std::vector<std::vector<Tensor>>;
    // Each inner vector is the specs from one branch
    using spec_return_value_t = std::vector<std::vector<TensorSpec>>;
    using program_factory_t = std::variant<ParallelProgramFactory>;

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

    // Hash together all branch hashes
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Launch the parallel device operation
std::vector<std::vector<Tensor>> parallel(const ttnn::experimental::prim::ParallelParams& operation_attributes);

}  // namespace ttnn::prim
