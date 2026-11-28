// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "block_max_bf16.hpp"

#include <algorithm>
#include <array>
#include <limits>

#include "device/block_max_bf16_device_operation.hpp"

namespace ttnn {

Tensor block_max_bf16(
    const Tensor& block_max,
    uint32_t factor,
    std::optional<uint32_t> n,
    const std::optional<Tensor>& n_tensor,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreCoord>& max_grid,
    bool strip_major) {
    using operations::block_max_bf16::BlockMaxBf16Params;
    // The caller's cap as given (the Python key held it raw); the factory clamps it to the device grid.
    std::optional<std::array<uint32_t, 2>> grid_cap;
    if (max_grid.has_value()) {
        constexpr size_t kMaxSide = std::numeric_limits<uint32_t>::max();
        grid_cap = std::array<uint32_t, 2>{
            static_cast<uint32_t>(std::min<size_t>(max_grid->x, kMaxSide)),
            static_cast<uint32_t>(std::min<size_t>(max_grid->y, kMaxSide))};
    }
    const BlockMaxBf16Params params{
        .strip_major = strip_major,
        .factor = factor,
        .n = n.value_or(0u),
        .max_grid = grid_cap,
        .output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
    };
    // The framework allocates the output before it validates, so the Python's assertions run here first.
    operations::block_max_bf16::validate_inputs(
        params, {.input = block_max, .n_tensor = n_tensor, .output = output_tensor}, n.has_value());
    return ttnn::prim::block_max_bf16(block_max, n_tensor, output_tensor, params);
}

}  // namespace ttnn
