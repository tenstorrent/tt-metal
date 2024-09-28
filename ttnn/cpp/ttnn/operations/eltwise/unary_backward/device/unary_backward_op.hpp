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
    ADD_BW,
    EQ_BW,
    GT_BW,
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
    RELU6_BW,
    SELU_BW,
    SQUARE_BW,
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
std::vector<Tensor> _relu6_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _selu_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _square_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);

std::vector<Tensor> _sub_bw( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _gt_bw( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config);

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
