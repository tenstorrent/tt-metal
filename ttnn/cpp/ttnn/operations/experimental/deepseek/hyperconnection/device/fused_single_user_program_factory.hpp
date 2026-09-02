// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "fused_single_user_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

struct FusedSingleUserSharedVariables {
    tt::tt_metal::KernelHandle collapse_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle post_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle comb_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle collapse_compute_kernel_id = 0;
    tt::tt_metal::KernelHandle post_compute_kernel_id = 0;
    tt::tt_metal::KernelHandle comb_compute_kernel_id = 0;
    tt::tt_metal::KernelHandle post_writer_kernel_id = 0;
    tt::tt_metal::KernelHandle comb_writer_kernel_id = 0;
    tt::tt_metal::CBHandle hidden_cb = 0;
    tt::tt_metal::CBHandle collapsed_output_cb = 0;
    std::vector<CoreCoord> collapse_cores;
    CoreCoord post_core;
    CoreCoord comb_core;
};

struct FusedSingleUserProgramFactory {
    using shared_variables_t = FusedSingleUserSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FusedSingleUserParams& operation_attributes,
        const FusedSingleUserInputs& tensor_args,
        FusedSingleUserTensorReturn& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FusedSingleUserParams& operation_attributes,
        const FusedSingleUserInputs& tensor_args,
        FusedSingleUserTensorReturn& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
