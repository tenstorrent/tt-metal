// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "layernorm_post_all_gather_device_operation_types.hpp"
#include "layernorm_post_all_gather_welford_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// Shared variables for normal program factory
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
        const LayerNormPostAllGatherParams& operation_attributes,
        const LayerNormPostAllGatherInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const LayerNormPostAllGatherParams& operation_attributes,
        const LayerNormPostAllGatherInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::prim
