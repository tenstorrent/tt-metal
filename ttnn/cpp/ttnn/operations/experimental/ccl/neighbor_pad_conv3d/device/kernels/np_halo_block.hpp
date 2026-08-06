// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

// A conv3d output block is a halo_last "boundary" block when any
inline bool np_is_boundary_block(
    uint32_t h_block, uint32_t h_block_end, uint32_t w_block, uint32_t w_block_end, uint32_t H_out, uint32_t W_out) {
    return (h_block == 0u) || (h_block_end >= H_out) || (w_block == 0u) || (w_block_end >= W_out);
}
