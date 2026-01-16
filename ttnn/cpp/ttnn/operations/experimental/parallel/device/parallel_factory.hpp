// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "parallel_device_operation_types.hpp"

namespace ttnn::operations::experimental::parallel {

// =============================================================================
// Shared Variables for Program Caching
// =============================================================================

struct ParallelSharedVariables {
    // Per-branch shared variables are stored in each BranchDescriptor
    // This struct only holds parallel-level state if needed

    // Number of branches (for validation during override)
    size_t num_branches = 0;
};

// =============================================================================
// Parallel Program Factory
// =============================================================================

struct ParallelProgramFactory {
    using shared_variables_t = ParallelSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    /**
     * Create a combined program from all branches
     *
     * This merges programs from all branches into a single program that can
     * execute them in parallel on their respective core ranges.
     */
    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    /**
     * Override runtime arguments for all branches
     */
    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::parallel
