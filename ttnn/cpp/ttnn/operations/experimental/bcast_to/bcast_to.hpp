// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ttnn/decorators.hpp"
namespace ttnn::operations::experimental {
struct BcastTo {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input,
        const Shape& output_shape,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& output);
};
}  // namespace ttnn::operations::experimental

namespace ttnn::experimental {
constexpr auto broadcast_to =
    ttnn::register_operation<"ttnn::experimental::broadcast_to", ttnn::operations::experimental::BcastTo>();
}
