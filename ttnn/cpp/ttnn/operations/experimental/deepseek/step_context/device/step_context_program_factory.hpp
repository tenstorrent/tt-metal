// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "step_context_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::step_context {

struct StepContextSharedVariables {
    tt::tt_metal::KernelHandle int_kernel_id = 0;
    tt::tt_metal::KernelHandle rope_kernel_id = 0;
    CoreCoord int_core;
    std::vector<CoreCoord> rope_cores;
    // Index of each integer output's address in the int core's runtime args.
    std::vector<uint32_t> int_addr_slots;
};

struct StepContextProgramFactory {
    using shared_variables_t = StepContextSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const StepContextParams& operation_attributes,
        const StepContextInputs& tensor_args,
        StepContextTensorReturn& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const StepContextParams& operation_attributes,
        const StepContextInputs& tensor_args,
        StepContextTensorReturn& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek::step_context
