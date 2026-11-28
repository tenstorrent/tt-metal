// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pack_fp8_rows.hpp"

#include <algorithm>

#include "device/pack_fp8_rows_device_operation.hpp"

namespace ttnn {

Tensor pack_fp8_rows(
    const Tensor& e4m3,
    const Tensor& scale,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::CoreCoord>& max_grid) {
    using operations::pack_fp8_rows::PackFp8RowsParams;
    PackFp8RowsParams params{
        .output_memory_config = output_tensor.has_value() ? output_tensor->memory_config()
                                                          : memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
    };
    // The framework allocates the output before it validates: refuse before anything is allocated. The Python
    // asserts ran before its first e4m3.device() use, so they come before the device check below.
    operations::pack_fp8_rows::check_support(params, {.e4m3 = e4m3, .scale = scale, .output = output_tensor});
    // Python _grid(): the device's compute grid, cut to max_grid (each at least 1). A deallocated or host tensor
    // has no device: Python's e4m3.device() call fails there too.
    auto* device = e4m3.device();
    TT_FATAL(device != nullptr, "pack_fp8_rows: e4m3 must be an allocated device tensor");
    const CoreCoord g = device->compute_with_storage_grid_size();
    params.grid_x = g.x;
    params.grid_y = g.y;
    if (max_grid.has_value()) {
        params.grid_x = std::max<uint32_t>(1, std::min<uint32_t>(g.x, max_grid->x));
        params.grid_y = std::max<uint32_t>(1, std::min<uint32_t>(g.y, max_grid->y));
    }
    return ttnn::prim::pack_fp8_rows(e4m3, scale, output_tensor, params);
}

}  // namespace ttnn
