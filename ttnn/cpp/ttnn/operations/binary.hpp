
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
