// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>

#include <tt_stl/span.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::expand {
struct ExpandOperation {
    static Tensor invoke(
        const ttnn::Tensor& input,
        const tt::stl::Span<const int32_t> shape_vector,
        const std::optional<MemoryConfig>& memory_config,
        const QueueId& queue_id = DefaultQueueId);
};
}  // namespace ttnn::operations::expand

namespace ttnn {
constexpr auto expand = ttnn::register_operation<"ttnn::expand", ttnn::operations::expand::ExpandOperation>();
}
