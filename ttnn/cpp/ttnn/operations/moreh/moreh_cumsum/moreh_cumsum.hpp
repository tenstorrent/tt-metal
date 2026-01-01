// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {

[[deprecated("moreh_cumsum is deprecated and will be removed after 2026-11-26. Use ttnn::cumsum instead.")]]
Tensor moreh_cumsum(
    const Tensor& input,
    int64_t dim,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor moreh_cumsum_backward(
    const Tensor& output_grad,
    int64_t dim,
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
