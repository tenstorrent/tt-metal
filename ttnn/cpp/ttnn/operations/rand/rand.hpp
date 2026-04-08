// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor rand(
    const ttnn::Shape& shape,
    MeshDevice& device,
    DataType dtype = DataType::BFLOAT16,
    Layout layout = Layout::TILE,
    const MemoryConfig& memory_config = types::DRAM_MEMORY_CONFIG,
    float from = 0.0f,
    float to = 1.0f,
    uint32_t seed = 0);

}  // namespace ttnn
