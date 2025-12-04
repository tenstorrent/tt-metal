// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "layernorm_post_all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <vector>

namespace ttnn::operations::normalization::layernorm_post_all_gather::program {

struct LayerNormPostAllGather2DSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores;
};

struct LayerNormPostAllGather2DProgramFactory {
    using shared_variables_t = LayerNormPostAllGather2DSharedVariables;
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

}  // namespace ttnn::operations::normalization::layernorm_post_all_gather::program
