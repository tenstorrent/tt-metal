// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "block_keys.hpp"

#include <algorithm>

#include "device/block_keys_device_operation.hpp"

namespace ttnn {

Tensor block_keys(
    const Tensor& pos,
    const Tensor& ids,
    uint32_t block,
    uint32_t width,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreCoord>& max_grid) {
    using operations::block_keys::BlockKeysParams;
    // Python _grid: the worker grid, clamped by max_grid (at least 1 x 1). A deallocated or host pos has no device
    // (the Python raised on its None device too): refuse instead of dereferencing it.
    const auto* device = pos.device();
    TT_FATAL(device != nullptr, "block_keys: pos must be an allocated device tensor");
    const CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t grid_x = grid.x;
    uint32_t grid_y = grid.y;
    if (max_grid.has_value()) {
        grid_x = std::max<uint32_t>(1u, std::min<uint32_t>(grid.x, max_grid->x));
        grid_y = std::max<uint32_t>(1u, std::min<uint32_t>(grid.y, max_grid->y));
    }
    const BlockKeysParams params{
        .block = block,
        .nbw = block == 0 ? 0u : width / block,  // block 0 is refused below, before any use
        .grid_x = grid_x,
        .grid_y = grid_y,
        // A preallocated output keeps its own placement (Python checks that one); else memory_config or DRAM.
        .output_memory_config = output_tensor.has_value() ? output_tensor->memory_config()
                                                          : memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
    };
    // The framework allocates the output before validating, so the Python's asserts run here first.
    operations::block_keys::validate_inputs(params, {.pos = pos, .ids = ids, .output = output_tensor});
    return ttnn::prim::block_keys(pos, ids, output_tensor, params);
}

}  // namespace ttnn
