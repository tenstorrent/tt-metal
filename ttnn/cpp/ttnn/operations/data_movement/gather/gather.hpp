// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"

namespace ttnn::operations::gather {
struct ExecuteGather {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        int8_t dim,
        const Tensor& input_index_tensor,
        bool sparse_grad,
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

}  // namespace ttnn::operations::gather

namespace ttnn {

constexpr auto gather = ttnn::register_operation<"ttnn::gather", ttnn::operations::gather::ExecuteGather>();

}  // namespace ttnn
