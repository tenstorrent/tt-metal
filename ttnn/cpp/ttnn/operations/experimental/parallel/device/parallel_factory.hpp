// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "parallel_device_operation_types.hpp"

namespace ttnn::experimental::prim {

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

    // Each inner vector is the outputs from one branch
    using tensor_return_value_t = std::vector<std::vector<Tensor>>;

    /**
     * Create a combined program from all branches
     *
     * This merges programs from all branches into a single program that can
     * execute them in parallel on their respective core ranges.
     */
    static cached_program_t create(
        const ParallelParams& operation_attributes,
        const ParallelInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    /**
     * Override runtime arguments for all branches
     */
    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ParallelParams& operation_attributes,
        const ParallelInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
