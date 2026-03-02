// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_gpt_fused_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::moe_gpt_fused::program {

struct MoEGPTFusedSharedVariables {
    std::map<std::string, tt::tt_metal::CBHandle> cb_handles_sharded;
    std::vector<tt::tt_metal::KernelHandle> kernel_handles;
    std::vector<CoreCoord> matmul_cores;
    std::vector<CoreCoord> gather_cores;
    std::vector<CoreCoord> combine_cores;
    std::shared_ptr<tt::tt_metal::Buffer> l1_input_buffer;  // Backing buffer for globally-allocated c_1
};

struct MoEGPTFusedProgramFactory {
    using shared_variables_t = MoEGPTFusedSharedVariables;
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

}  // namespace ttnn::operations::experimental::moe_gpt_fused::program
