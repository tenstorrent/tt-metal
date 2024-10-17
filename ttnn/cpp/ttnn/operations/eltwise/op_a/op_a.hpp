// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"


namespace ttnn {

namespace operations {

namespace op_a {

struct ExecuteUnaryWithFloatParameter {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const float parameter = 0,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        const float parameter = 0,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace op_a
}  // namespace operations

constexpr auto op_a = ttnn::register_operation_with_auto_launch_op<
        "ttnn::op_a",
        ttnn::operations::op_a::ExecuteUnaryWithFloatParameter>();

}   // namespace ttnn
