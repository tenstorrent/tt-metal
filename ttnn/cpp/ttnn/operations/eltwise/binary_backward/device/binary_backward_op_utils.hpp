// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>

#include <tt-metalium/base_types.hpp>
#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/shape/shape.hpp"
#include "binary_backward_op_types.hpp"

namespace ttnn::operations::binary_backward {

// One row per migrated op. Adding a gradient is one row plus one compute kernel.
struct BinaryBackwardKernelSpec {
    std::string_view compute_kernel_path;
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    // Force UnpackToDestFp32 + fp32_dest_acc_en regardless of I/O dtype; halves DEST
    // slots (8 -> 4 for bfloat16), so enable only when the kernel needs it.
    bool force_fp32_dest_acc = false;
    // Fused multiply+reduce compute kernel for broadcast cases. Empty here: broadcast rows
    // use the composite reduce_to_shape path in binary_backward.cpp; follow-up wires this.
    std::string_view broadcast_reduce_kernel_path;
};

const BinaryBackwardKernelSpec& kernel_spec(BinaryBackwardOpType op_type);

// Sum-reduce axes to recover `operand_shape` from `grad_shape`. Right-aligns shapes
// (standard broadcast rule); precondition: grad_shape.rank() >= operand_shape.rank().
ttsl::SmallVector<int64_t> broadcast_reduce_axes(const ttnn::Shape& operand_shape, const ttnn::Shape& grad_shape);

// True when `grad_shape` broadcasts strictly beyond `operand_shape` (i.e. any
// diverging axis exists). Used by the composite path to gate reduce_to_shape.
bool is_broadcasted_over(const ttnn::Shape& operand_shape, const ttnn::Shape& grad_shape);

}  // namespace ttnn::operations::binary_backward
