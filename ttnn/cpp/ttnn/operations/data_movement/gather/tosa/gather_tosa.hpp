// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"

namespace ttnn::operations::data_movement {
struct ExecuteTosaGather {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const Tensor& input_index_tensor,
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::data_movement

namespace ttnn::tosa {

constexpr auto gather =
    ttnn::register_operation<"ttnn::tosa_gather", ttnn::operations::data_movement::ExecuteTosaGather>();

}  // namespace ttnn::tosa
