// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// N-slice-outer K split (perf experiment): the order in which a core of a grid row computes and sends the
// row's owner N-slices. Compute, writer and reader walk the same order, so each slice CB's FIFO order matches
// its sender's. Step j of core `my_col` handles owner `nslice_owner(my_col, j)` for j in [0, k_cols); owners
// >= num_owning_cols (columns past the last C_out tile) are skipped. The core's own slice comes last: it only
// needs a local write, so the remote slices' sends all overlap a later matmul.
//   order 0 (rotate):    my_col + 1, my_col + 2, ..., my_col (mod k_cols)
//   order 1 (alternate): my_col + 1, my_col - 1, my_col + 2, my_col - 2, ..., my_col (mod k_cols)

#pragma once

#include <cstdint>

namespace conv2d_3x3 {

template <uint32_t k_cols, uint32_t order>
inline uint32_t nslice_owner(uint32_t my_col, uint32_t j) {
    if constexpr (order == 0) {
        return (my_col + 1 + j) % k_cols;
    } else {
        if (j == k_cols - 1) {
            return my_col;
        }
        const uint32_t d = j / 2 + 1;  // distance
        return (j % 2 == 0) ? (my_col + d) % k_cols : (my_col + k_cols - d) % k_cols;
    }
}

// Sender of step j's owner slice y: mode 0 = writer (all), 1 = writer for own and lower columns / reader for
// higher, 2 = reader (all), 3 = by direction under the alternate order: the +d steps (even j) on the reader's NoC0,
// the -d steps (odd j) and the own slice on the writer's NoC1, so every core sends about half on each NoC and each
// slice goes the way its NoC runs.
template <uint32_t mode, uint32_t k_cols>
inline bool nslice_reader_sends(uint32_t my_col, uint32_t y, uint32_t j) {
    if constexpr (mode == 3) {
        return j != k_cols - 1 && j % 2 == 0;
    }
    return mode == 2 || (mode == 1 && y > my_col);
}

}  // namespace conv2d_3x3
