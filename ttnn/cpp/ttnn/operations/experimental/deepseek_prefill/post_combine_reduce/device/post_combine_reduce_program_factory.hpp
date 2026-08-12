// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "post_combine_reduce_types.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce {

struct PostCombineReduceProgramFactory {
    // Per-coord program build.  The post_combine_reduce kernels are coord-invariant
    // (no fabric, no per-device row index), so this factory uses the no-coord
    // overload and the framework replicates the same descriptor across the mesh.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PostCombineReduceParams& operation_attributes,
        const PostCombineReduceInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce
