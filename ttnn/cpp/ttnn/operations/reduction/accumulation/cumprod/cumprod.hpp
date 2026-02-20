// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor cumprod(
    const Tensor& input_tensor,
    const int32_t& dim,
    std::optional<DataType> dtype = std::nullopt,
    const bool& reverse_order = false,
    std::optional<Tensor> optional_out = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
