// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fused_pre_post_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

struct FusedPrePostSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
};

struct FusedPrePostProgramFactory {
    using shared_variables_t = FusedPrePostSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FusedPrePostParams& operation_attributes,
        const FusedPrePostInputs& tensor_args,
        FusedPrePostTensorReturn& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FusedPrePostParams& operation_attributes,
        const FusedPrePostInputs& tensor_args,
        FusedPrePostTensorReturn& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
