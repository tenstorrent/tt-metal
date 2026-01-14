// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::data_movement::concat::program {

// Shared variables for s2s multi-tensor concat
struct ConcatS2SMultiSharedVariables {
    uint32_t num_input_tensors = 0;
    std::vector<tt::tt_metal::CBHandle> cb_inputs;
    tt::tt_metal::CBHandle cb_output = 0;
    CoreRangeSet all_cores;
};

struct ConcatS2SMultiProgramFactory {
    using shared_variables_t = ConcatS2SMultiSharedVariables;
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

}  // namespace ttnn::operations::data_movement::concat::program
