// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/types.hpp"

namespace ttnn {

ttnn::Tensor roll(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int>& shifts,
    const ttsl::SmallVector<int>& input_dims,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

ttnn::Tensor roll(
    const ttnn::Tensor& input_tensor, int shifts, const std::optional<MemoryConfig>& memory_config = std::nullopt);

ttnn::Tensor roll(
    const ttnn::Tensor& input_tensor,
    int shifts,
    int dim,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
