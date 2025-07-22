// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ttnn/decorators.hpp"

namespace ttnn::operations::expand {
struct ExpandOperation {
    static Tensor invoke(
        const ttnn::Tensor& input,
        tt::stl::Span<const int32_t> shape_vector,
        const std::optional<MemoryConfig>& memory_config,
        const QueueId& queue_id = DefaultQueueId);
};
}  // namespace ttnn::operations::expand

namespace ttnn {
constexpr auto expand = ttnn::register_operation<"ttnn::expand", ttnn::operations::expand::ExpandOperation>();
}
