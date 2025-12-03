// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "untilize_device_operation_types.hpp"

namespace ttnn::operations::data_movement::program {

struct UntilizeMultiCoreParallelizeColumnProgramFactory {
    using shared_variables_t = untilize::program::untilize_shared_variables_t;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const untilize::operation_attributes_t& operation_attributes,
        const untilize::tensor_args_t& tensor_args,
        const untilize::tensor_return_value_t& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const untilize::operation_attributes_t& operation_attributes,
        const untilize::tensor_args_t& tensor_args,
        const untilize::tensor_return_value_t& tensor_return_value);
};
}  // namespace ttnn::operations::data_movement::program
