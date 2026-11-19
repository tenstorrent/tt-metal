// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 weight server: the DRAM page of a round's tile, shared by every engine that reads a share of a round
// (the server reader on NoC0, the server writer on NoC1, helper cores).

#pragma once

#include <cstdint>

namespace conv2d_3x3 {

// Round kb's tile t. The K-block's tiles are consecutive pages (bn is the whole C_out on this path); kw-fold's round of
// column kw reads the prepared tile ((kh * 3 + kw) C_in + c, n) for slot (kh, c, n) (round kb is n-group kb / 3 of rw
// output column tiles, weight column kw = (1, 0, 2)[kb % 3]).
template <bool kw_fold, uint32_t block_k_tiles, uint32_t block_n_tiles, uint32_t tensor_n_tiles>
FORCE_INLINE uint32_t server_page_of(uint32_t kb, uint32_t t) {
    if constexpr (kw_fold) {
        // Guarded: this branch is compiled on every shape (block_n_tiles may be < 3 off kw-fold).
        constexpr uint32_t rw = block_n_tiles >= 3 ? block_n_tiles / 3 : 1;
        const uint32_t n = t % rw;
        const uint32_t kc = t / rw;
        const uint32_t kh = kc / block_k_tiles;
        const uint32_t col = kb % 3;
        const uint32_t kw = col == 0 ? 1 : (col == 1 ? 0 : 2);  // kw_fold.hpp KW_ROUND_ORDER
        return ((kh * 3 + kw) * block_k_tiles + (kc - kh * block_k_tiles)) * tensor_n_tiles + (kb / 3) * rw + n;
    } else {
        return kb * block_k_tiles * block_n_tiles + t;
    }
}

}  // namespace conv2d_3x3
