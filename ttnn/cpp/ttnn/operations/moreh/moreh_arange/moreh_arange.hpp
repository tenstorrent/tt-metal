// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {

Tensor moreh_arange(
    double start,
    double end,
    double step,
    ttnn::MeshDevice* mesh_device,
    const std::optional<Tensor>& output = std::nullopt,
    bool untilize_out = false,
    const DataType& dtype = DataType::BFLOAT16,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG);

}  // namespace ttnn
