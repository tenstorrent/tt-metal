// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/operations/eltwise/binary_backward/device/binary_backward_op.hpp"

namespace ttnn::operations::unary_backward {

enum class UnaryBackwardOpType {
    DIV_BW,
    MULTIGAMMALN_BW,
    ADD_BW,
    EQ_BW,
    GT_BW,
    LGAMMA_BW,
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
    RELU_BW,
    LOGIT_BW,
    FLOOR_BW,
    RELU6_BW,
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
    LOG2_BW,
    SIGN_BW,
    EXP2_BW,
    EXPM1_BW,
    DIGAMMA_BW,
    ERFINV_BW,
    ERF_BW,
    DEG2RAD_BW,
};

std::vector<Tensor> _acos_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _atan_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _rad2deg_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _frac_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _trunc_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _log_sigmoid_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _fill_zero_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _i0_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _tan_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _sigmoid_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _ceil_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _softsign_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _cosh_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _log2_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _sign_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _exp2_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _expm1_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _digamma_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _erfinv_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _erf_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _deg2rad_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _hardswish_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _tanhshrink_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _atanh_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _asin_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _asinh_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _sin_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _sinh_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _log10_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _log1p_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _erfc_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _relu6_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _selu_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _square_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);

std::vector<Tensor> _sub_bw( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _gt_bw( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config);

std::vector<Tensor> _multigammaln_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _lgamma_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _hardsigmoid_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _cos_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _acosh_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _relu_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _logit_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _floor_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);

std::vector<Tensor> _add_bw( const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
std::vector<Tensor> _eq_bw( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& output_mem_config);

// OpHandler struct template
template <UnaryBackwardOpType OpType>
struct OpHandler;

template <>
struct OpHandler<UnaryBackwardOpType::ACOS_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _acos_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::ATAN_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _atan_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::RAD2DEG_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _rad2deg_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::FRAC_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _frac_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::TAN_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _tan_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SIGMOID_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _sigmoid_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::CEIL_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _ceil_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SOFTSIGN_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _softsign_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::COSH_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _cosh_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::LOG2_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _log2_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SIGN_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _sign_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::EXP2_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _exp2_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::EXPM1_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _expm1_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::DIGAMMA_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _digamma_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::ERFINV_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _erfinv_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::ERF_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _erf_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::DEG2RAD_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _deg2rad_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::HARDSWISH_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _hardswish_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::TANHSHRINK_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _tanhshrink_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::ATANH_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _atanh_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::ASIN_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _asin_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::ASINH_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _asinh_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SIN_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _sin_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SINH_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _sinh_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::LOG10_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _log10_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::LOG1P_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _log1p_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::ERFC_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _erfc_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::TRUNC_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _trunc_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::LOG_SIGMOID_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _log_sigmoid_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::FILL_ZERO_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _fill_zero_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::I0_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _i0_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::RELU6_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _relu6_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SELU_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _selu_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SQUARE_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
        return _square_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::GT_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _gt_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::MULTIGAMMALN_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _multigammaln_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::LGAMMA_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _lgamma_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::HARDSIGMOID_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _hardsigmoid_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::COS_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _cos_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::ACOSH_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _acosh_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::RELU_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _relu_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::LOGIT_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _logit_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::FLOOR_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _floor_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SUB_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config ) {
        return _sub_bw(grad, input, scalar, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::EQ_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _eq_bw(grad, input, other, output_mem_config);
    }
};

}  // namespace ttnn::operations::unary_backward
