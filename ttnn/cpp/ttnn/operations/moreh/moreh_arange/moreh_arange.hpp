// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor moreh_arange(
    float start,
    float end,
    float step,
    ttnn::MeshDevice* mesh_device,
    const std::optional<Tensor>& output = std::nullopt,
    bool untilize_out = false,
    const DataType& dtype = DataType::BFLOAT16,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG);

}  // namespace ttnn
