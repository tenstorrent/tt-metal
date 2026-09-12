// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include <tt-metalium/base_types.hpp>

#include "unary_backward_op_types.hpp"

namespace ttnn::operations::unary_backward {

// Everything that differs between two unary-backward gradients sharing this device operation.
// The shared program factory reads this and nothing else, so adding an op is one table row
// plus one compute kernel -- no new device operation, program factory or validation.
struct UnaryBackwardKernelSpec {
    // Repo-relative path to the compute kernel. It receives grad_output on c_0, input on c_1
    // and writes the input gradient to c_2, with per_core_tile_cnt in runtime arg 0.
    std::string_view compute_kernel_path;

    // Kernels that keep intermediates in DEST need HiFi4 to be worth fusing; a gradient whose
    // math is a plain multiply can drop to a cheaper fidelity by saying so here.
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;

    // Accumulate in float32 DEST even when every operand is narrower. Set this for a gradient
    // whose intermediates cancel -- the fused kernel keeps them in DEST, so it can hold them at
    // float32 where the composite had to round each one back to the operand dtype in L1. Costs
    // half the DEST slots (8 -> 4 for bfloat16), so it is per-op rather than blanket.
    bool force_fp32_dest_acc = false;
};

const UnaryBackwardKernelSpec& get_kernel_spec(UnaryBackwardOpType op_type);

// Op name for TT_FATAL messages, so shared validation can still name the op the user called.
std::string_view to_string(UnaryBackwardOpType op_type);

}  // namespace ttnn::operations::unary_backward
