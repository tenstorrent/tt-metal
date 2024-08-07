// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteUntilize {
    static ttnn::Tensor operator()(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor,
        const std::optional<MemoryConfig> &memory_config,
        bool use_multicore = true,
        bool use_pack_untilize = true);

    static ttnn::Tensor operator()(
        const ttnn::Tensor &input_tensor,
        const std::optional<MemoryConfig> &memory_config,
        bool use_multicore = true,
        bool use_pack_untilize = true);
};

}  // namespace operations::data_movement

constexpr auto untilize =
    ttnn::register_operation_with_auto_launch_op<"ttnn::untilize", ttnn::operations::data_movement::ExecuteUntilize>();

}  // namespace ttnn
