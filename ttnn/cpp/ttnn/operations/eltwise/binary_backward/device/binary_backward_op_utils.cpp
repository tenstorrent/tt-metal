// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_backward_op_utils.hpp"

#include <algorithm>
#include <cstddef>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::binary_backward {

namespace {

constexpr BinaryBackwardKernelSpec kMulBw{
    .compute_kernel_path = "ttnn/cpp/ttnn/operations/eltwise/binary_backward/device/kernels/compute/eltwise_bw_mul.cpp",
    .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
    .force_fp32_dest_acc = false,
    .broadcast_reduce_kernel_path = {},
};

}  // namespace

const BinaryBackwardKernelSpec& kernel_spec(BinaryBackwardOpType op_type) {
    switch (op_type) {
        case BinaryBackwardOpType::MUL_BW: return kMulBw;
    }
    TT_THROW("binary_backward: unknown BinaryBackwardOpType={}", static_cast<uint32_t>(op_type));
}

ttsl::SmallVector<int64_t> broadcast_reduce_axes(const ttnn::Shape& operand_shape, const ttnn::Shape& grad_shape) {
    const auto operand_rank = operand_shape.rank();
    const auto grad_rank = grad_shape.rank();
    TT_FATAL(
        operand_rank <= grad_rank,
        "binary_backward: operand rank {} exceeds grad rank {}; a well-formed backward call has "
        "grad_output.rank() == forward_output.rank() >= max(operand_rank)",
        operand_rank,
        grad_rank);

    const auto rank_diff = grad_rank - operand_rank;
    ttsl::SmallVector<int64_t> axes;
    axes.reserve(grad_rank);
    for (size_t i = 0; i < grad_rank; ++i) {
        const uint32_t operand_dim = (i < rank_diff) ? 1u : operand_shape[i - rank_diff];
        const uint32_t grad_dim = grad_shape[i];
        if (operand_dim != grad_dim) {
            axes.push_back(static_cast<int64_t>(i));
        }
    }
    return axes;
}

bool is_broadcasted_over(const ttnn::Shape& operand_shape, const ttnn::Shape& grad_shape) {
    if (operand_shape.rank() > grad_shape.rank()) {
        return false;
    }
    // Rank divergence alone still needs a reshape even if no dim diverges (e.g. operand
    // (32,128) vs grad (1,1,32,128)); treat it as "broadcasted over" for gate purposes.
    if (operand_shape.rank() != grad_shape.rank()) {
        return true;
    }
    return !broadcast_reduce_axes(operand_shape, grad_shape).empty();
}

}  // namespace ttnn::operations::binary_backward
