// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dit_layernorm_post_all_gather_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::transformer::dit_layernorm::program {

struct PostAllGatherWelfordSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct PostAllGatherWelfordProgramFactory {
    using shared_variables_t = PostAllGatherWelfordSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const PostAllGatherOperationAttributes& operation_attributes,
        const PostAllGatherTensorArgs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const PostAllGatherOperationAttributes& operation_attributes,
        const PostAllGatherTensorArgs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::experimental::transformer::dit_layernorm::program
