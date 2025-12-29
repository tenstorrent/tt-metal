// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "layernorm_post_all_gather_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::normalization::program {

// Shared variables for non-2D program factories (normal and Welford)
struct LayerNormPostAllGatherSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

// Program factory for normal (non-Welford) operation
struct LayerNormPostAllGatherProgramFactory {
    using shared_variables_t = LayerNormPostAllGatherSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const LayerNormPostAllGatherOperationAttributes& operation_attributes,
        const LayerNormPostAllGatherTensorArgs& tensor_args,
        LayerNormPostAllGatherTensorReturnValue& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const LayerNormPostAllGatherOperationAttributes& operation_attributes,
        const LayerNormPostAllGatherTensorArgs& tensor_args,
        LayerNormPostAllGatherTensorReturnValue& tensor_return_value);
};

// Program factory for Welford algorithm (layernorm only)
struct LayerNormPostAllGatherWelfordProgramFactory {
    using shared_variables_t = LayerNormPostAllGatherSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const LayerNormPostAllGatherOperationAttributes& operation_attributes,
        const LayerNormPostAllGatherTensorArgs& tensor_args,
        LayerNormPostAllGatherTensorReturnValue& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const LayerNormPostAllGatherOperationAttributes& operation_attributes,
        const LayerNormPostAllGatherTensorArgs& tensor_args,
        LayerNormPostAllGatherTensorReturnValue& tensor_return_value);
};

}  // namespace ttnn::operations::normalization::program
