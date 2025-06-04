// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/decorators.hpp"

#include <ranges>

namespace ttnn {
namespace operations {
namespace data_movement {

struct ConcatOperation {
    // Wrapper for TTDNN
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const std::vector<ttnn::Tensor>& input_tensors,
        int dim,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
        unsigned int groups = 1);
};

}  // namespace data_movement
}  // namespace operations

constexpr auto concat = ttnn::register_operation<"ttnn::concat", ttnn::operations::data_movement::ConcatOperation>();

}  // namespace ttnn
