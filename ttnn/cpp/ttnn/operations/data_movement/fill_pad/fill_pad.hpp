// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
Tensor fill_implicit_tile_padding(
    const Tensor& input_tensor, float fill_value, const std::optional<MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn
