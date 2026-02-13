// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

namespace ttnn::experimental {
Tensor broadcast_to(
    const Tensor& input,
    const Shape& output_shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output);
}  // namespace ttnn::experimental
