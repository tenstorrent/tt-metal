// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include <optional>

namespace ttnn::operations::data_movement {

struct ExecuteSort {
    static std::vector<Tensor> invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        int8_t dim,
        bool descending,
        bool stable,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<std::tuple<Tensor&, Tensor&>> optional_output_tensors = std::nullopt);
};

}  // namespace ttnn::operations::data_movement

namespace ttnn {

constexpr auto sort = ttnn::register_operation<"ttnn::sort", ttnn::operations::data_movement::ExecuteSort>();

}  // namespace ttnn
