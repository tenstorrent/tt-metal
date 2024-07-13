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
    HARDTANH_BW,
    THRESHOLD_BW,
    SOFTPLUS_BW,
    ASSIGN_BW,
    MULTIGAMMALN_BW,
    ADD_BW,
    EQ_BW,
    GT_BW,
    LT_BW,
    LE_BW,
    GE_BW,
    NE_BW,
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
    ASINH_BW,
    SIN_BW,
    SINH_BW,
    LOG10_BW,
    LOG1P_BW,
    ERFC_BW,
    CEIL_BW,
    SOFTSIGN_BW,
    COSH_BW,
    LOGITEPS_BW,
    LOG2_BW,
    SIGN_BW,
    FMOD_BW,
    REMAINDER_BW,
    DIV_NO_NAN_BW,
    EXP2_BW,
    EXPM1_BW,
    RECIPROCAL_BW,
    DIGAMMA_BW,
    ERFINV_BW,
    ERF_BW,
    DEG2RAD_BW,
};

struct UnaryBackwardFunction{
    static std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const MemoryConfig&)> get_function_type1(UnaryBackwardOpType OpType);
    static std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, float, const MemoryConfig&)> get_function_type1_w_float(UnaryBackwardOpType OpType);
};

//OpHandler_two_float : get_function_type1_w_two_float
std::vector<Tensor> _clamp_bw( const Tensor& grad, const Tensor& input, float min, float max, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _hardtanh_bw( const Tensor& grad, const Tensor& input, float min, float max, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _threshold_bw( const Tensor& grad, const Tensor& input, float threshold, float value, const std::optional<MemoryConfig>& output_mem_config);

//OpHandler_two_float_with_default : get_function_type1_w_two_float_with_default
std::vector<Tensor> _softplus_bw( const Tensor& grad, const Tensor& input, float beta = 1.0, float threshold = 20.0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

// OpHandler struct template
template <UnaryBackwardOpType OpType>
struct OpHandler_two_float;

// OpHandler struct template
template <UnaryBackwardOpType OpType>
struct OpHandler_two_float_with_default;

template <>
struct OpHandler_two_float<UnaryBackwardOpType::CLAMP_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float min, float max, const std::optional<MemoryConfig>& output_mem_config ) {
        return _clamp_bw(grad, input, min, max, output_mem_config);
    }
};

template <>
struct OpHandler_two_float<UnaryBackwardOpType::HARDTANH_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float min, float max, const std::optional<MemoryConfig>& output_mem_config ) {
        return _hardtanh_bw(grad, input, min, max, output_mem_config);
    }
};

template <>
struct OpHandler_two_float<UnaryBackwardOpType::THRESHOLD_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float threshold, float value, const std::optional<MemoryConfig>& output_mem_config ) {
        return _threshold_bw(grad, input, threshold, value, output_mem_config);
    }
};

template <>
struct OpHandler_two_float_with_default<UnaryBackwardOpType::SOFTPLUS_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float beta, float threshold, const std::optional<MemoryConfig>& output_mem_config ) {
        return _softplus_bw(grad, input, beta, threshold, output_mem_config);
    }
};

// Template functions to get the function pointers
template <UnaryBackwardOpType OpType>
auto get_function_type1_w_two_float() {
    return &OpHandler_two_float<OpType>::handle;
}

template <UnaryBackwardOpType OpType>
auto get_function_type1_w_two_float_with_default() {
    return &OpHandler_two_float_with_default<OpType>::handle;
}

}  // namespace ttnn::operations::unary_backward
