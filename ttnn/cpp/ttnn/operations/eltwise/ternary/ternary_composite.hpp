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
struct ExecuteTernaryCompositeOps
{
    static Tensor operator()(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const Tensor& input_tensor_c,
        float value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        auto op_type = get_ternary_fn_float<ternary_comp_op_type>();
        return op_type(input_tensor_a, input_tensor_b, input_tensor_c, value, memory_config);
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

}  // namespace ttnn
