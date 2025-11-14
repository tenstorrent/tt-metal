// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "clip_grad_norm_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::clip_grad_norm::program {

struct ClipGradNormSharedVariables {
    tt::tt_metal::KernelHandle clip_grad_norm_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle clip_grad_norm_writer_kernel_id = 0;
    tt::tt_metal::KernelHandle clip_grad_norm_kernel_group_1_id = 0;
    tt::tt_metal::KernelHandle clip_grad_norm_kernel_group_2_id = 0;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
};

struct ClipGradNormProgramFactory {
    using shared_variables_t = ClipGradNormSharedVariables;
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

}  // namespace ttnn::operations::experimental::clip_grad_norm::program
