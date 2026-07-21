// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "attn_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct AttnMatmulProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  cb_src0's total_size depends on Kt
    // (input shape), so padded_shape is folded into compute_program_hash() — each
    // unique Kt keeps its own cache entry.  On cache hit, the framework copies
    // runtime args and patches dynamic CB addresses but does NOT re-apply CB
    // total_size/page_size (the cached descriptor already carries them).
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AttnMatmulParams& operation_attributes, const AttnMatmulInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
