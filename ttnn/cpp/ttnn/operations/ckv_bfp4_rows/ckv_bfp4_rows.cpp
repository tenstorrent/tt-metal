// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ckv_bfp4_rows.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>

#include "device/ckv_bfp4_rows_device_operation.hpp"

namespace ttnn {

void ckv_bfp4_rows(
    const Tensor& src,
    const Tensor& rows,
    uint32_t src_row0,
    uint32_t n,
    uint32_t row0,
    const std::optional<CoreCoord>& max_grid,
    const std::optional<Tensor>& output_tensor) {
    using namespace operations::ckv_bfp4_rows;
    // The Python's asserts, in its order, before anything is allocated or launched.
    check_support(src, rows, src_row0, n, row0);
    if (n == 0) {
        return;
    }
    const CoreCoord grid = src.device()->compute_with_storage_grid_size();
    uint32_t gx = grid.x;
    uint32_t gy = grid.y;
    if (max_grid.has_value()) {
        gx = std::max<uint32_t>(1u, std::min<uint32_t>(grid.x, max_grid->x));
        gy = std::max<uint32_t>(1u, std::min<uint32_t>(grid.y, max_grid->y));
    }
    const CkvBfp4RowsParams params{
        .src_row0 = src_row0,
        .num_bands = n / tt::constants::TILE_HEIGHT,
        .row0 = row0,
        .grid_x = gx,
        .grid_y = gy,
    };
    ttnn::prim::ckv_bfp4_rows(src, rows, output_tensor, params);
}

}  // namespace ttnn
