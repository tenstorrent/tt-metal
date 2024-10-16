// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteUntilize {
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const ttnn::Tensor &input_tensor,
                               const std::optional<MemoryConfig> &memory_config = std::nullopt,
                               bool use_multicore = true,
                               bool use_pack_untilize = true);

    static ttnn::Tensor invoke(const ttnn::Tensor &input_tensor,
                               const std::optional<MemoryConfig> &memory_config = std::nullopt,
                               bool use_multicore = true,
                               bool use_pack_untilize = true);
};

}  // namespace operations::data_movement

constexpr auto untilize =
    ttnn::register_operation_with_auto_launch_op<"ttnn::untilize", ttnn::operations::data_movement::ExecuteUntilize>();

}  // namespace ttnn
