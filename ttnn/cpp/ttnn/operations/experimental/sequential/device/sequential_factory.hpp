// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "sequential_device_operation_types.hpp"

namespace ttnn::operations::experimental::sequential {

// =============================================================================
// Shared Variables for Program Caching
// =============================================================================

struct SequentialSharedVariables {
    // Number of steps (for validation during override)
    size_t num_steps = 0;
    // Each step stores its own core range
};

// =============================================================================
// Sequential Program Factory
// =============================================================================

struct SequentialProgramFactory {
    using shared_variables_t = SequentialSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    /**
     * Create a combined program from all steps
     *
     * This adds all steps to a single program, executing them in sequence
     * on the same set of cores.
     */
    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    /**
     * Override runtime arguments for all steps
     */
    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::sequential
