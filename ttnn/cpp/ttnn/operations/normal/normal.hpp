// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor normal(
    const ttnn::Shape& shape,
    MeshDevice& device,
    DataType dtype = DataType::BFLOAT16,
    Layout layout = Layout::TILE,
    const MemoryConfig& memory_config = types::DRAM_MEMORY_CONFIG,
    float mean = 0.0f,
    float stddev = 1.0f,
    std::optional<uint32_t> seed = std::nullopt);

}  // namespace ttnn
