// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "layernorm_pre_all_gather_device_operation_types.hpp"
#include "layernorm_pre_all_gather_welford_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// Shared variables for 1D program factory (normal)
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
        const LayerNormPreAllGatherParams& operation_attributes, const Tensor& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const LayerNormPreAllGatherParams& operation_attributes,
        const Tensor& tensor_args,
        Tensor& output);
};

// Program factory for 2D core grid operation
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
