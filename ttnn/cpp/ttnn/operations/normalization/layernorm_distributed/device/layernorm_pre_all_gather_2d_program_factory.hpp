// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "layernorm_pre_all_gather_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/kernel_types.hpp>

namespace ttnn::prim {

// Shared variables for 2D program factory
struct LayerNormPreAllGather2DSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = {};
    tt::tt_metal::KernelHandle writer_kernel_id = {};
    tt::tt_metal::KernelHandle compute_kernel_id = {};
    uint32_t cores_x = 0;
    uint32_t cores_y = 0;
};

// 2D program factory
struct LayerNormPreAllGather2DProgramFactory {
    using shared_variables_t = LayerNormPreAllGather2DSharedVariables;
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
