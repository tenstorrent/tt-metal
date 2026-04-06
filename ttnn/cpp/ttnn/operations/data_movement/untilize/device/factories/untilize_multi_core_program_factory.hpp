// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "../untilize_device_operation_types.hpp"

namespace ttnn::prim {

struct UntilizeMultiCoreProgramFactory {
    using shared_variables_t = UntilizeSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const UntilizeOperationAttributes& operation_attributes,
        const UntilizeTensorArgs& tensor_args,
        const UntilizeTensorReturnValue& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const UntilizeOperationAttributes& operation_attributes,
        const UntilizeTensorArgs& tensor_args,
        const UntilizeTensorReturnValue& tensor_return_value);
};
}  // namespace ttnn::prim
