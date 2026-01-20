// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "layernorm_pre_all_gather_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// Shared variables for Welford program factory
struct LayerNormPreAllGatherWelfordSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    uint32_t num_cores = 0;
    CoreCoord grid_size;
};

// Program factory for Welford algorithm (layernorm only)
struct LayerNormPreAllGatherWelfordProgramFactory {
    using shared_variables_t = LayerNormPreAllGatherWelfordSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const LayerNormPreAllGatherParams& operation_attributes, const Tensor& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const LayerNormPreAllGatherParams& operation_attributes,
        const Tensor& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::prim
