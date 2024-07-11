// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::unary_backward {

constexpr uint8_t DefaultQueueId = 0;
enum class UnaryBackwardOpType {
    MUL_BW,
    CLAMP_MIN_BW,
    CLAMP_BW,
    ASSIGN_BW,
    MULTIGAMMALN_BW,
    ADD_BW,
    EQ_BW,
    LGAMMA_BW,
    FILL_BW,
    HARDSIGMOID_BW,
    COS_BW,
    ACOSH_BW,
    ACOS_BW,
    ATAN_BW,
    RAD2DEG_BW,
    SUB_BW,
    FRAC_BW,
    TRUNC_BW,
    LOG_SIGMOID_BW,
    FILL_ZERO_BW,
    I0_BW,
    TAN_BW,
    SIGMOID_BW,
    RSQRT_BW,
    NEG_BW,
    RELU_BW,
    LOGIT_BW,
    CLAMP_MAX_BW,
    HARDSHRINK_BW,
    SOFTSHRINK_BW,
    LEAKY_RELU_BW,
    ELU_BW,
    CELU_BW,
    RPOW_BW,
    FLOOR_BW,
    ROUND_BW,
    LOG_BW,
    RELU6_BW,
    ABS_BW,
    SILU_BW,
    SELU_BW,
    SQUARE_BW,
    HARDSWISH_BW,
    TANHSHRINK_BW,
    ATANH_BW,
    ASIN_BW,
    ASINH_BW
    SIN_BW,
    SINH_BW,
    LOG10_BW,
    LOG1P_BW,
    ERFC_BW,
    CEIL_BW,
};

struct UnaryBackwardFunction{
    static std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const MemoryConfig&)> get_function_type1(UnaryBackwardOpType OpType);
    static std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, float, const MemoryConfig&)> get_function_type1_w_float(UnaryBackwardOpType OpType);
    static std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, float, float, const MemoryConfig&)> get_function_type1_w_two_float(UnaryBackwardOpType OpType);
};

}  // namespace ttnn::operations::unary_backward
