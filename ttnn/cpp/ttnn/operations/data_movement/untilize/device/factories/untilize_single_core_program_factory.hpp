// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation_types.hpp"
namespace ttnn::operations::data_movement::program {

struct UntilizeSingleCoreProgramFactory {
    using shared_variables_t = ttnn::operations::data_movement::untilize_types::program::untilize_shared_variables_t;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ttnn::operations::data_movement::untilize_types::operation_attributes_t& operation_attributes,
        const ttnn::operations::data_movement::untilize_types::tensor_args_t& tensor_args,
        const ttnn::operations::data_movement::untilize_types::tensor_return_value_t& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ttnn::operations::data_movement::untilize_types::operation_attributes_t& operation_attributes,
        const ttnn::operations::data_movement::untilize_types::tensor_args_t& tensor_args,
        const ttnn::operations::data_movement::untilize_types::tensor_return_value_t& tensor_return_value);
};
}  // namespace ttnn::operations::data_movement::program
