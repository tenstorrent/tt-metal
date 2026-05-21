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
    // (KV_HEADS, Mt, Kt, Nt) computed from input shapes and are intentionally kept
    // out of the program hash — see hash_cb_descriptor in program_descriptors.cpp.
    // On cache hit, apply_descriptor_runtime_args re-applies CB sizes (PR #44939)
    // so a single cached program absorbs shape variation cheaply — same scheme as
    // the legacy override_runtime_arguments + UpdateCircularBufferTotalSize path.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GroupAttnMatmulParams& operation_attributes,
        const GroupAttnMatmulInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
