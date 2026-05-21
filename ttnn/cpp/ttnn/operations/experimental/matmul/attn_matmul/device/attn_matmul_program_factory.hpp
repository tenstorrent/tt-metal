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
    // (input shape) and is deliberately kept out of the program hash — so a single
    // cached program absorbs Kt variation cheaply via the framework's slow-path
    // CB-size re-application (see apply_descriptor_runtime_args).
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AttnMatmulParams& operation_attributes, const AttnMatmulInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
