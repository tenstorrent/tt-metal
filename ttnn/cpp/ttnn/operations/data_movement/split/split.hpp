// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/types.hpp"

namespace ttnn {

std::vector<ttnn::Tensor> split(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<int64_t>& split_sizes,
    int64_t dim = 0,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

std::vector<ttnn::Tensor> split(
    const ttnn::Tensor& input_tensor,
    int64_t split_size,
    int64_t dim = 0,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
