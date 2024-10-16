// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <optional>

namespace ttnn {
namespace operations::data_movement {

struct MoveOperation {
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const Tensor& input_tensor,
                               const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

    static ttnn::Tensor invoke(const Tensor& input_tensor,
                               const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto move = ttnn::register_operation<"ttnn::move", ttnn::operations::data_movement::MoveOperation>();

}  // namespace ttnn
