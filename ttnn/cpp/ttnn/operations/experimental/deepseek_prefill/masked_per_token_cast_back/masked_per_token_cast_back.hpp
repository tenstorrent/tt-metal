// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::masked_per_token_cast_back {

ttnn::Tensor masked_per_token_cast_back(
    const ttnn::Tensor& input_e4m3,
    const std::optional<ttnn::Tensor>& input_scale,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t experts_per_chip,
    const std::optional<tt::tt_metal::DataType>& output_dtype = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& metadata = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::masked_per_token_cast_back

namespace ttnn {
using operations::experimental::deepseek_prefill::masked_per_token_cast_back::masked_per_token_cast_back;
}  // namespace ttnn
