// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fill_rm_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::data_movement::fill_rm::program {

struct FillRMSharedVariables {
    tt::tt_metal::KernelHandle kernel_id = 0;
};

struct FillRMProgramFactory {
    using shared_variables_t = FillRMSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FillRmParams& operation_attributes, const FillRmInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FillRmParams& operation_attributes,
        const FillRmInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::data_movement::fill_rm::program
