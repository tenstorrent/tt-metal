// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_composite_op.hpp"

namespace ttnn {

namespace operations {

namespace ternary {

template <TernaryCompositeOpType ternary_comp_op_type>
struct ExecuteTernaryCompositeOps
{
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const Tensor& input_tensor_c,
        float value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_type0<ternary_comp_op_type>();
            return op_type(input_tensor_a, input_tensor_b, input_tensor_c, value, memory_config);
        }
};

}  // namespace ternary
}  // namespace operations

// newly imported
constexpr auto addcmul = ttnn::register_operation<operations::ternary::ExecuteTernaryCompositeOps<operations::ternary::TernaryCompositeOpType::ADDCMUL>>("ttnn::addcmul");
constexpr auto addcdiv = ttnn::register_operation<operations::ternary::ExecuteTernaryCompositeOps<operations::ternary::TernaryCompositeOpType::ADDCDIV>>("ttnn::addcdiv");


}  // namespace ttnn
