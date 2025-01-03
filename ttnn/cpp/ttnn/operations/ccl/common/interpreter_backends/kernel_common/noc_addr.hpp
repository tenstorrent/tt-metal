// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include <cstdint>

// NOTE: This will eventually be updated with an official API
static constexpr size_t VIRTUAL_COORDS_START_X = 16;
static constexpr size_t VIRTUAL_COORDS_START_Y = 16;
FORCE_INLINE bool is_using_noc_coords(uint16_t noc_x, uint16_t noc_y) {
    return noc_x < VIRTUAL_COORDS_START_X && noc_y < VIRTUAL_COORDS_START_Y;
}

FORCE_INLINE uint64_t safe_get_noc_addr(uint8_t dest_noc_x, uint8_t dest_noc_y, uint32_t dest_bank_addr) {
    bool using_noc_coords = is_using_noc_coords(dest_noc_x, dest_noc_y);
    uint8_t noc_x = dest_noc_x;
    uint8_t noc_y = dest_noc_y;
    if (using_noc_coords) {
        noc_x = NOC_X_PHYS_COORD(dest_noc_x);
        noc_y = NOC_Y_PHYS_COORD(dest_noc_y);
    }
    return get_noc_addr(noc_x, noc_y, dest_bank_addr);
}
