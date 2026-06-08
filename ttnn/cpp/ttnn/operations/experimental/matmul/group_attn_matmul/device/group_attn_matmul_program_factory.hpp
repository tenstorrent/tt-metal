// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "group_attn_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct GroupAttnMatmulProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  Several CB total_sizes depend on
    // (KV_HEADS, Mt, Kt, Nt) computed from input shapes, so padded_shape is folded
    // into compute_program_hash() to keep each unique CB sizing in its own cache
    // entry.  On cache hit, apply_descriptor_runtime_args copies runtime args and
    // patches dynamic CB addresses; CB total_size/page_size are not re-applied
    // (the cached descriptor already carries the correct values).
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GroupAttnMatmulParams& operation_attributes,
        const GroupAttnMatmulInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
