// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pair_pool.hpp"

#include <algorithm>

#include "device/pair_pool_device_operation.hpp"

namespace ttnn {

Tensor pair_pool(
    const Tensor& x,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreCoord>& max_grid) {
    using operations::pair_pool::PairPoolParams;
    const PairPoolParams probe{};
    const operations::pair_pool::PairPoolInputs inputs{.input = x, .output = output_tensor};
    // The Python asserts ran before the output was allocated; the framework allocates before it validates.
    operations::pair_pool::check_support(probe, inputs);

    const CoreCoord g = x.device()->compute_with_storage_grid_size();
    uint32_t gx = g.x;
    uint32_t gy = g.y;
    if (max_grid.has_value()) {
        gx = std::max<uint32_t>(1, std::min<uint32_t>(g.x, max_grid->x));
        gy = std::max<uint32_t>(1, std::min<uint32_t>(g.y, max_grid->y));
    }
    MemoryConfig out_mc = ttnn::DRAM_MEMORY_CONFIG;
    if (output_tensor.has_value()) {
        out_mc = output_tensor->memory_config();
    } else if (memory_config.has_value()) {
        out_mc = *memory_config;
    }
    const PairPoolParams params{.grid_x = gx, .grid_y = gy, .output_memory_config = out_mc};
    return ttnn::prim::pair_pool(x, output_tensor, params);
}

}  // namespace ttnn
