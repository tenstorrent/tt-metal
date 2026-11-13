// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_heads_decode(
    const ttnn::Tensor& input_tensor,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental::llama31_8b
