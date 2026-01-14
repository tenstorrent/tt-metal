// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_device_operation_types.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_common.hpp"

namespace ttnn::operations::data_movement::repeat::program {

struct RepeatProgramFactoryHigherDim {
    using shared_variables_t = RepeatSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::data_movement::repeat::program
