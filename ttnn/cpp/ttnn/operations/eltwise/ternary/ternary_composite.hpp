// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/core.hpp"
#include "ternary_composite_op.hpp"
namespace ttnn {

namespace operations::ternary {

template <TernaryCompositeOpType ternary_comp_op_type>
struct ExecuteTernaryCompositeOps {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const Tensor& input_tensor_c,
        float value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<ternary_comp_op_type>::handle(
            input_tensor_a, input_tensor_b, input_tensor_c, value, memory_config);
    }
};

template <TernaryCompositeOpType ternary_comp_op_type>
struct ExecuteTernaryCompositeLerp {
    static Tensor invoke(
        const Tensor& input_tensor,
        const Tensor& end,
        const Tensor& weight,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<ternary_comp_op_type>::handle(input_tensor, end, weight, memory_config);
    }

    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        float value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<ternary_comp_op_type>::handle(input_tensor_a, input_tensor_b, value, memory_config);
    }
};

inline Tensor mac(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& input_tensor_c,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return OpHandler<TernaryCompositeOpType::MAC>::handle(
        input_tensor_a, input_tensor_b, input_tensor_c, memory_config);
}

inline Tensor mac(
    const Tensor& input_tensor_a,
    float value1,
    float value2,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return OpHandler<TernaryCompositeOpType::MAC>::handle(input_tensor_a, value1, value2, memory_config);
}

}  // namespace operations::ternary

using operations::ternary::mac;

}  // namespace ttnn
