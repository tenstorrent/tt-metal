// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_gpt_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::moe_gpt::program {

struct MoEGPTSharedVariables {
    // CB handles for sharded circular buffers
    std::map<std::string, tt::tt_metal::CBHandle> cb_handles_sharded;

    // Kernel handles (matmul)
    std::vector<tt::tt_metal::KernelHandle> kernel_handles;

    // Kernel handles (tilize)
    std::vector<tt::tt_metal::KernelHandle> tilize_kernel_handles;

    // Kernel handle (combine dm1, fused mode only)
    std::optional<tt::tt_metal::KernelHandle> combine_kernel_handle;

    // Matmul cores
    std::vector<CoreCoord> worker_cores;

    // Tilize cores
    std::vector<CoreCoord> tilize_cores;

    // Combine cores (fused mode only)
    std::vector<CoreCoord> combine_cores;
};

struct MoEGPTProgramFactory {
    using shared_variables_t = MoEGPTSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::moe_gpt::program
