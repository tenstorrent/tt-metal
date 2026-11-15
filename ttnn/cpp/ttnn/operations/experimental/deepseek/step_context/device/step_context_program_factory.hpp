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
    // The integer outputs are split over int_cores in order; int_addr_slots[c] is the runtime-arg index of each of
    // core c's outputs' address, its first output int_first[c].
    std::vector<CoreCoord> int_cores;
    std::vector<uint32_t> int_first;
    std::vector<std::vector<uint32_t>> int_addr_slots;
    // One core per RoPE row; rope_output[i] is core i's RoPE output.
    std::vector<CoreCoord> rope_cores;
    std::vector<uint32_t> rope_output;
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
