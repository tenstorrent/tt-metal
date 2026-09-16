// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>

#include <tt-metalium/base_types.hpp>

#include "binary_backward_op_types.hpp"

namespace ttnn::operations::binary_backward {

// One row per migrated op. Adding a gradient is one row plus one compute kernel.
struct BinaryBackwardKernelSpec {
    std::string_view compute_kernel_path;
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    // Force UnpackToDestFp32 + fp32_dest_acc_en regardless of I/O dtype. Halves
    // available DEST slots (8 -> 4 for bfloat16); enable only when the kernel
    // needs it.
    bool force_fp32_dest_acc = false;
    // Number of gradients the kernel emits. MUL_BW = 2 (input_grad, other_grad).
    uint8_t num_outputs = 2;
};

const BinaryBackwardKernelSpec& kernel_spec(BinaryBackwardOpType op_type);

}  // namespace ttnn::operations::binary_backward
