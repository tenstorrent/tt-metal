// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor select_final_carry(
    const ttnn::Tensor& rank_final,
    const ttnn::Tensor& prefix_final,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::optional<ttnn::Tensor>& actual_end = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    uint32_t sequence_parallel_axis = 0,
    std::optional<uint32_t> num_links = std::nullopt);

}  // namespace ttnn::experimental::kda
