// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "layernorm_pre_all_gather_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::normalization::program {

// Shared variables for 1D program factories (normal and Welford)
struct LayerNormPreAllGatherSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    uint32_t num_cores = 0;
    CoreCoord grid_size;
};

// Shared variables for 2D program factory
struct LayerNormPreAllGather2DSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

// Program factory for normal (non-Welford, non-2D) operation
struct LayerNormPreAllGatherProgramFactory {
    using shared_variables_t = LayerNormPreAllGatherSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const LayerNormPreAllGatherOperationAttributes& operation_attributes,
        const LayerNormPreAllGatherTensorArgs& tensor_args,
        LayerNormPreAllGatherTensorReturnValue& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const LayerNormPreAllGatherOperationAttributes& operation_attributes,
        const LayerNormPreAllGatherTensorArgs& tensor_args,
        LayerNormPreAllGatherTensorReturnValue& tensor_return_value);
};

// Program factory for 2D core grid operation
struct LayerNormPreAllGather2DProgramFactory {
    using shared_variables_t = LayerNormPreAllGather2DSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const LayerNormPreAllGatherOperationAttributes& operation_attributes,
        const LayerNormPreAllGatherTensorArgs& tensor_args,
        LayerNormPreAllGatherTensorReturnValue& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const LayerNormPreAllGatherOperationAttributes& operation_attributes,
        const LayerNormPreAllGatherTensorArgs& tensor_args,
        LayerNormPreAllGatherTensorReturnValue& tensor_return_value);
};

// Program factory for Welford algorithm (layernorm only)
struct LayerNormPreAllGatherWelfordProgramFactory {
    using shared_variables_t = LayerNormPreAllGatherSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const LayerNormPreAllGatherOperationAttributes& operation_attributes,
        const LayerNormPreAllGatherTensorArgs& tensor_args,
        LayerNormPreAllGatherTensorReturnValue& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const LayerNormPreAllGatherOperationAttributes& operation_attributes,
        const LayerNormPreAllGatherTensorArgs& tensor_args,
        LayerNormPreAllGatherTensorReturnValue& tensor_return_value);
};

}  // namespace ttnn::operations::normalization::program
