// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// Shared variables for s2s tiled concat
struct ConcatS2STiledSharedVariables {
    uint32_t num_input_tensors = 0;
    std::vector<tt::tt_metal::CBHandle> cb_inputs;
    tt::tt_metal::CBHandle cb_output = 0;
    CoreRangeSet all_cores;
};

struct ConcatS2STiledProgramFactory {
    using shared_variables_t = ConcatS2STiledSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value);
    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ConcatParams& operation_attributes,
        const ConcatInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
