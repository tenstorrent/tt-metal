// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_grid_utils.hpp"

#include <algorithm>
#include <cmath>

namespace ttnn::operations::normalization {

uint32_t compute_num_virtual_cols(uint32_t grid_x, int num_groups, uint32_t num_channels) {
    uint32_t nvc = std::min<uint32_t>(grid_x, num_groups);
    while (nvc > 0 && ((num_channels / nvc) % ttnn::types::TILE_SIZE != 0 || (num_groups % nvc) != 0)) {
        nvc -= 1;
    }
    return nvc;
}

std::optional<ttnn::CoreGrid> find_expected_dram_grid(
    uint32_t max_x, uint32_t max_y, uint32_t num_channels, int num_groups, uint32_t input_nhw) {
    uint32_t Ht = static_cast<uint32_t>(std::ceil(static_cast<double>(input_nhw) / ttnn::types::TILE_SIZE));

    for (uint32_t gx = max_x; gx >= 1; --gx) {
        uint32_t nvc = compute_num_virtual_cols(gx, num_groups, num_channels);
        if (nvc == 0) {
            continue;
        }
        uint32_t rows_per_y = gx / nvc;
        if (rows_per_y == 0) {
            continue;
        }
        uint32_t max_gy = std::min<uint32_t>(Ht / rows_per_y, max_y);
        for (uint32_t gy = max_gy; gy >= 1; --gy) {
            uint32_t num_virtual_rows = rows_per_y * gy;
            if (Ht % num_virtual_rows == 0) {
                return ttnn::CoreGrid(gx, gy);
            }
        }
    }
    return std::nullopt;
}

}  // namespace ttnn::operations::normalization
