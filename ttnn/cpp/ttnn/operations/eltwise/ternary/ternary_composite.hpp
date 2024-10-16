// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ternary_composite_op.hpp"

namespace ttnn {

namespace operations {

namespace ternary {

template <TernaryCompositeOpType ternary_comp_op_type>
struct ExecuteTernaryCompositeOps {
    static Tensor invoke(const Tensor& input_tensor_a,
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
    static Tensor invoke(const Tensor& input_tensor_a,
                         const Tensor& input_tensor_b,
                         const Tensor& input_tensor_c,
                         const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<ternary_comp_op_type>::handle(input_tensor_a, input_tensor_b, input_tensor_c, memory_config);
    }

    static Tensor invoke(const Tensor& input_tensor_a,
                         const Tensor& input_tensor_b,
                         float value,
                         const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<ternary_comp_op_type>::handle(input_tensor_a, input_tensor_b, value, memory_config);
    }
};

template <TernaryCompositeOpType ternary_comp_op_type>
struct ExecuteTernaryCompositeMac {
    static Tensor invoke(const Tensor& input_tensor_a,
                         const Tensor& input_tensor_b,
                         const Tensor& input_tensor_c,
                         const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<ternary_comp_op_type>::handle(input_tensor_a, input_tensor_b, input_tensor_c, memory_config);
    }

    static Tensor invoke(const Tensor& input_tensor_a,
                         float value1,
                         float value2,
                         const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<ternary_comp_op_type>::handle(input_tensor_a, value1, value2, memory_config);
    }
};

}  // namespace ternary
}  // namespace operations

constexpr auto addcmul = ttnn::register_operation_with_auto_launch_op<
    "ttnn::addcmul",
    operations::ternary::ExecuteTernaryCompositeOps<operations::ternary::TernaryCompositeOpType::ADDCMUL>>();
constexpr auto addcdiv = ttnn::register_operation_with_auto_launch_op<
    "ttnn::addcdiv",
    operations::ternary::ExecuteTernaryCompositeOps<operations::ternary::TernaryCompositeOpType::ADDCDIV>>();
constexpr auto lerp = ttnn::register_operation_with_auto_launch_op<
    "ttnn::lerp",
    operations::ternary::ExecuteTernaryCompositeLerp<operations::ternary::TernaryCompositeOpType::LERP>>();
constexpr auto mac = ttnn::register_operation_with_auto_launch_op<
    "ttnn::mac",
    operations::ternary::ExecuteTernaryCompositeMac<operations::ternary::TernaryCompositeOpType::MAC>>();

}  // namespace ttnn
