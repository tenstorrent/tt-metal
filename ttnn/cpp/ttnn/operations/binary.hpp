
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/op_library/binary/binary_op.hpp"

namespace ttnn {

constexpr auto add = ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::ADD, false>>("ttnn::add");
constexpr auto add_ = ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::ADD, true>>("ttnn::add_");
constexpr auto subtract =
    ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::SUB, false>>("ttnn::subtract");
constexpr auto subtract_ =
    ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::SUB, true>>("ttnn::subtract_");
constexpr auto multiply =
    ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::MUL, false>>("ttnn::multiply");
constexpr auto multiply_ =
    ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::MUL, true>>("ttnn::multiply_");

constexpr auto eq = ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::EQ, false>>("ttnn::eq");
constexpr auto ne = ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::NE, false>>("ttnn::ne");
constexpr auto ge = ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::GTE, false>>("ttnn::ge");
constexpr auto gt = ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::GT, false>>("ttnn::gt");
constexpr auto le = ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::LTE, false>>("ttnn::le");
constexpr auto lt = ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::LT, false>>("ttnn::lt");
constexpr auto logical_and =
    ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::LOGICAL_AND, false>>("ttnn::logical_and");
constexpr auto logical_or =
    ttnn::register_operation<ttnn::operations::binary::Binary<BinaryOpType::LOGICAL_OR, false>>("ttnn::logical_or");

template <typename InputBType>
ttnn::Tensor operator+(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return add(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator-(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return subtract(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator*(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return multiply(input_tensor_a, scalar);
}

}  // namespace ttnn
