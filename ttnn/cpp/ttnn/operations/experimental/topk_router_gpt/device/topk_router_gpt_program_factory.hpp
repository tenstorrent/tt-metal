// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "topk_router_gpt_device_operation_types.hpp"
#include "tt_metal/api/tt-metalium/program.hpp"

namespace ttnn::operations::experimental::topk_router_gpt {

struct TopkRouterGptProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle dm0_kernel_id;
        tt::tt_metal::KernelHandle dm1_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        CoreRangeSet all_cores;
        uint32_t num_cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);
};

}  // namespace ttnn::operations::experimental::topk_router_gpt
