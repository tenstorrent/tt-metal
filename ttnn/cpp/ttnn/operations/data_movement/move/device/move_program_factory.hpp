// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "move_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_program_factory.hpp"

namespace ttnn::operations::data_movement::move::program {

// Program factory for MULTI_CORE and MULTI_CORE_OVERLAP strategies
struct MoveProgramFactory {
    using shared_variables_t = ttnn::operations::data_movement::copy::program::CopySharedVariables;
    using cached_program_t = ttnn::operations::data_movement::copy::program::CopyProgramFactory::cached_program_t;

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

}  // namespace ttnn::operations::data_movement::move::program
