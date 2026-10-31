// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_backward_op_utils.hpp"

#include <algorithm>
#include <cstddef>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::binary_backward {

namespace {

// String-literal concat keeps paths constexpr; matches binary_ng/ternary utils style.
#define BW_COMPUTE(file) "ttnn/cpp/ttnn/operations/eltwise/binary_backward/device/kernels/compute/" file

using Spec = BinaryBackwardKernelSpec;

// 1-input ops: ∂y/∂a, ∂y/∂b are constants (±1); kernel reads grad only (see program factory).
constexpr Spec kAddBw{.compute_kernel_path = BW_COMPUTE("eltwise_bw_add.cpp"), .num_input_tensors = 1};
constexpr Spec kSubBw{.compute_kernel_path = BW_COMPUTE("eltwise_bw_sub.cpp"), .num_input_tensors = 1};
constexpr Spec kRsubBw{.compute_kernel_path = BW_COMPUTE("eltwise_bw_rsub.cpp"), .num_input_tensors = 1};

// 3-input ops: ∂y/∂? uses operand values; kernel reads grad + input + other.
constexpr Spec kMulBw{.compute_kernel_path = BW_COMPUTE("eltwise_bw_mul.cpp")};
// div_bw: Recip + Square need fp32 dest acc for 1/other² accuracy.
constexpr Spec kDivBw{.compute_kernel_path = BW_COMPUTE("eltwise_bw_div.cpp"), .force_fp32_dest_acc = true};

#undef BW_COMPUTE

}  // namespace

const BinaryBackwardKernelSpec& kernel_spec(BinaryBackwardOpType op_type) {
    switch (op_type) {
        case BinaryBackwardOpType::MUL_BW: return kMulBw;
        case BinaryBackwardOpType::ADD_BW: return kAddBw;
        case BinaryBackwardOpType::SUB_BW: return kSubBw;
        case BinaryBackwardOpType::RSUB_BW: return kRsubBw;
        case BinaryBackwardOpType::DIV_BW: return kDivBw;
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
        TT_FATAL(
            operand_dim == 1u || operand_dim == grad_dim,
            "binary_backward: operand shape {} not broadcastable to grad shape {} at axis {} "
            "(operand_dim={}, grad_dim={}); expected operand_dim == 1 or == grad_dim.",
            operand_shape,
            grad_shape,
            i,
            operand_dim,
            grad_dim);
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
