// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteUntilizeWithUnpadding {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor,
        const ttnn::Shape &output_tensor_end,
        const std::optional<MemoryConfig> &memory_config,
        bool use_multicore = false,
        bool use_pack_untilize = true);

    static ttnn::Tensor invoke(
        const ttnn::Tensor &input_tensor,
        const ttnn::Shape &output_tensor_end,
        const std::optional<MemoryConfig> &memory_config,
        bool use_multicore = false,
        bool use_pack_untilize = true);
};

}  // namespace operations::data_movement

constexpr auto untilize_with_unpadding = ttnn::register_operation_with_auto_launch_op<
    "ttnn::untilize_with_unpadding",
    ttnn::operations::data_movement::ExecuteUntilizeWithUnpadding>();

}  // namespace ttnn
