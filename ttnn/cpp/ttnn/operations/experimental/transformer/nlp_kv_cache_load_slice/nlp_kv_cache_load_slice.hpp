// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

using tt::tt_metal::MemoryConfig;

namespace ttnn::experimental {

Tensor nlp_kv_cache_load_slice(
    const Tensor& input_tensor,
    uint32_t seq_len_start,
    uint32_t seq_len_end,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn::experimental
