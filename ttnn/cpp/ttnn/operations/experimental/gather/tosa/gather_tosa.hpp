// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::tosa::gather {
struct ExecuteTosaGather {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const Tensor& input_index_tensor,
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::experimental::tosa::gather

namespace ttnn::experimental::tosa {

constexpr auto gather = ttnn::register_operation<
    "ttnn::experimental::tosa_gather",
    ttnn::operations::experimental::tosa::gather::ExecuteTosaGather>();

}  // namespace ttnn::experimental::tosa
