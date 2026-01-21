// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "sequential_device_operation_types.hpp"

namespace ttnn::experimental::prim {

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

    // Sequential returns the outputs from the LAST step only
    using tensor_return_value_t = std::vector<Tensor>;

    /**
     * Create a combined program from all steps
     *
     * This adds all steps to a single program, executing them in sequence
     * on the same set of cores.
     */
    static cached_program_t create(
        const SequentialParams& operation_attributes,
        const SequentialInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    /**
     * Override runtime arguments for all steps
     */
    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const SequentialParams& operation_attributes,
        const SequentialInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
