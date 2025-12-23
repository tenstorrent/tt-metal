// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::reduction::moe {

struct ExecuteMoe {
    static Tensor invoke(
        const Tensor& input_tensor,
        const Tensor& expert_mask_tensor,
        const Tensor& topk_mask_tensor,
        uint16_t k,
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output_tensor = std::nullopt);
};

}  // namespace ttnn::operations::reduction::moe

namespace ttnn {

constexpr auto moe = ttnn::register_operation<"ttnn::moe", ttnn::operations::reduction::moe::ExecuteMoe>();

}  // namespace ttnn
