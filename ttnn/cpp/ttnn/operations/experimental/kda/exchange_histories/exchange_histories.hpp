// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

std::vector<ttnn::Tensor> exchange_histories(
    const ttnn::Tensor& projected,
    uint32_t width,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::optional<ttnn::Tensor>& actual_end = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    uint32_t sequence_parallel_axis = 0);

}  // namespace ttnn::experimental::kda
