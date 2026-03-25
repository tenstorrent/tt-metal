// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "topk_router_gpt_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::topk_router_gpt::program {

struct TopkRouterGptSharedVariables {
    // Kernel handles
    std::vector<tt::tt_metal::KernelHandle> kernel_handles;

    // Cores active
    std::vector<CoreCoord> worker_cores;
};

struct TopkRouterGptProgramFactory {
    using shared_variables_t = TopkRouterGptSharedVariables;
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

}  // namespace ttnn::operations::experimental::topk_router_gpt::program
