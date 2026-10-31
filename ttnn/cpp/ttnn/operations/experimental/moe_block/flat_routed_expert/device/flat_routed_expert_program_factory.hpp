// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flat_program.hpp"
#include "flat_routed_expert_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

// The plain op: one program, the same on every chip (the flat fragment alone). Cache hits re-apply the buffer
// addresses (the arena / words are per-call buffers, so they move between calls).
struct FlatRoutedExpertProgramFactory {
    using shared_variables_t = FlatProgramHandles;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FlatRoutedExpertParams& operation_attributes,
        const FlatRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FlatRoutedExpertParams& operation_attributes,
        const FlatRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
