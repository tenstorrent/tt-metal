// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/device_operation.hpp"
#include "parallel_device_operation_types.hpp"
#include "parallel_factory.hpp"

namespace ttnn::operations::experimental::parallel {

// =============================================================================
// ParallelDeviceOperation
// =============================================================================

struct ParallelDeviceOperation {
    using operation_attributes_t = parallel::operation_attributes_t;
    using tensor_args_t = parallel::tensor_args_t;
    using spec_return_value_t = parallel::spec_return_value_t;
    using tensor_return_value_t = parallel::tensor_return_value_t;
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
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::parallel

namespace ttnn::prim {

// Launch the parallel device operation
ttnn::operations::experimental::parallel::ParallelDeviceOperation::tensor_return_value_t parallel(
    const ttnn::operations::experimental::parallel::operation_attributes_t& operation_attributes);

}  // namespace ttnn::prim
