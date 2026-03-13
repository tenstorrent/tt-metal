// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

ttnn::Tensor moreh_full(
    const ttnn::SmallVector<uint32_t>& shape,
    std::variant<float, int> fill_value,
    ttnn::MeshDevice* mesh_device,
    const DataType& dtype = DataType::BFLOAT16,
    const Layout& layout = ttnn::TILE_LAYOUT,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG);

}  // namespace ttnn
