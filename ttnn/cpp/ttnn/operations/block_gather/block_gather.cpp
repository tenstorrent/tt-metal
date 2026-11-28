// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "block_gather.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>

#include "device/block_gather_device_operation.hpp"

namespace ttnn {

Tensor block_gather(
    const Tensor& scores,
    const Tensor& ids,
    int64_t block,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreCoord>& max_grid) {
    using operations::block_gather::BlockGatherParams;
    // Python: B = int(block); assert 1 <= B <= 32 and B & (B - 1) == 0, before B is narrowed to the kernel's uint32.
    TT_FATAL(
        block >= 1 && block <= 32 && std::has_single_bit(static_cast<uint64_t>(block)),
        "block_gather: block {} must be a power of two <= 32",
        block);
    BlockGatherParams params{
        .block = static_cast<uint32_t>(block),
        .output_memory_config = memory_config.has_value() ? *memory_config : ttnn::DRAM_MEMORY_CONFIG,
        .grid_x = 1,  // resolved below, once the device is known
        .grid_y = 1,
    };
    // The framework allocates the output before it validates: refuse here first, in the Python's order.
    operations::block_gather::validate_inputs(params, {.scores = scores, .ids = ids, .output = output_tensor});
    // Python _grid(scores.device(), max_grid): the device grid, each side clamped to max(1, min(grid, max_grid)). A
    // deallocated or host tensor has no device: the Python's scores.device() use fails there too.
    auto* device = scores.device();
    TT_FATAL(device != nullptr, "block_gather: scores must be an allocated device tensor");
    const CoreCoord grid = device->compute_with_storage_grid_size();
    params.grid_x = grid.x;
    params.grid_y = grid.y;
    if (max_grid.has_value()) {
        params.grid_x = std::max<uint32_t>(1u, std::min<uint32_t>(params.grid_x, max_grid->x));
        params.grid_y = std::max<uint32_t>(1u, std::min<uint32_t>(params.grid_y, max_grid->y));
    }
    return ttnn::prim::block_gather(scores, ids, output_tensor, params);
}

}  // namespace ttnn
