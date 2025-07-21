// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

FORCE_INLINE unsigned get_tile_id(
    uint32_t i0,
    uint32_t i1,
    uint32_t j,
    uint32_t tiles_per_row,
    uint32_t product_low_dims,
    uint32_t product_high_dims,
    uint32_t HtWt) {
    uint32_t base_tileid = i0 * (tiles_per_row * product_high_dims * HtWt) + i1;
    uint32_t tileid = base_tileid + j * product_high_dims * HtWt;
    return tileid;
}
