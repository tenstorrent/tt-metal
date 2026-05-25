// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk {

std::array<Tensor, 2> iterative_topk(
    const Tensor& input, uint32_t k, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk
