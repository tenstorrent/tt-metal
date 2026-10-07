// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::unary_backward {

// Gradients that run as a single fused device operation rather than as a composition of
// forward ops. One entry per op migrated off the composite path in unary_backward.cpp;
// each maps to a compute kernel through the table in unary_backward_op_utils.cpp.
//
// The enum is the program-cache discriminator: it is part of operation_attributes_t, so two
// op types can never share a cache entry even though they share this device operation.
enum class UnaryBackwardOpType : uint8_t {
    SIGMOID_BW,
    TANH_BW,
    CELU_BW,
    SELU_BW,
    LOG_SIGMOID_BW,
    SOFTPLUS_BW,
    HARDSIGMOID_BW,
    HARDTANH_BW,
    LEAKY_RELU_BW,
    RELU6_BW,
    HARDSHRINK_BW,
    SOFTSHRINK_BW,
    ABS_BW,
    ACOS_BW,
    ASIN_BW,
    ATANH_BW,
    ASINH_BW,
    LOGIT_BW,
    LOGITEPS_BW,
    SQRT_BW,
    RSQRT_BW,
    LOG_BW,
    LOG2_BW,
    LOG10_BW,
    LOG1P_BW,
    RECIPROCAL_BW,
    EXPM1_BW,
    EXP2_BW,
    SQUARE_BW,
    SINH_BW,
    COSH_BW,
    ERFINV_BW,
    MULTIGAMMALN_BW,
    DIGAMMA_BW,
};

}  // namespace ttnn::operations::unary_backward
