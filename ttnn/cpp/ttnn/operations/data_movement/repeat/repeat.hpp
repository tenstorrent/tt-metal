// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"


namespace ttnn {
namespace operations::data_movement {

struct RepeatOperation {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const Shape & repeat_dims,
        const std::optional<MemoryConfig>& memory_config_arg);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const Shape & repeat_dims,
        const std::optional<MemoryConfig>& memory_config);

    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const Shape & repeat_dims);
};


}  // namespace operations::data_movement

constexpr auto repeat = ttnn::register_operation_with_auto_launch_op<"ttnn::repeat", ttnn::operations::data_movement::RepeatOperation>();

}  // namespace ttnn
