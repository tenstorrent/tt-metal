// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// N-slice-outer K split: the order in which a core of a grid row computes and sends the row's owner N-slices, and
// which data-movement RISC sends each. Compute, writer and reader walk the same order, so each slice CB's FIFO order
// matches its sender's. Step j handles owner my_col + 1, my_col - 1, my_col + 2, my_col - 2, ... (mod k_cols), the
// core's own slice last: it is a local write, so every remote send overlaps a later slice's matmul. Owners past the
// last C_out tile are skipped.
// The +d steps go out on the reader's NoC0 (which runs toward higher columns), the -d steps and the own slice on the
// writer's NoC1, so each core sends about half its slices on each NoC, each in its NoC's direction.

#pragma once

#include <cstdint>

namespace conv2d_3x3 {

#ifndef NSD_ORDER
#define NSD_ORDER 0
#endif
// nslice_send_decouple experiment: NSD_ORDER 0 = committed (both RISCs nearest first), 1 = both RISCs farthest first,
// 2 = only the reader (NoC0, +d) farthest first, 3 = only the writer (NoC1, -d) farthest first.
template <uint32_t k_cols>
inline uint32_t nslice_owner(uint32_t my_col, uint32_t j) {
#ifdef NSD_RING
    // Ring reduce-scatter (nslice_send_decouple): step j handles owner my_col - 1 - j, so the partial for owner o
    // starts on core o + 1 and walks +1 column per step (NoC0, one hop) to o, which adds its own slice last.
    return (my_col + 2 * k_cols - 1 - j) % k_cols;
#endif
    if (j == k_cols - 1) {
        return my_col;
    }
    constexpr uint32_t reader_steps = k_cols / 2;        // +1 .. +k/2 (even j)
    constexpr uint32_t writer_steps = (k_cols - 1) / 2;  // -1 .. -(k-1)/2 (odd j)
    const uint32_t i = j / 2;
    uint32_t d = i + 1;
    if (j % 2 == 0) {
        if constexpr (NSD_ORDER == 1 || NSD_ORDER == 2) {
            d = reader_steps - i;
        }
    } else {
        if constexpr (NSD_ORDER == 1 || NSD_ORDER == 3) {
            d = writer_steps - i;
        }
    }
    return (j % 2 == 0) ? (my_col + d) % k_cols : (my_col + k_cols - d) % k_cols;
}

template <uint32_t k_cols>
inline bool nslice_reader_sends(uint32_t j) {
    return j != k_cols - 1 && j % 2 == 0;
}

// Ring reduce-scatter helpers (NSD_RING). A step is real when its owner owns output columns.
template <uint32_t k_cols, uint32_t num_owning_cols>
inline uint32_t ring_next_real(uint32_t my_col, uint32_t j) {
    for (; j < k_cols; ++j) {
        if (nslice_owner<k_cols>(my_col, j) < num_owning_cols) {
            return j;
        }
    }
    return k_cols;
}

// Real steps j >= 1 of one M-block: each takes exactly one incoming partial from column my_col - 1.
template <uint32_t k_cols, uint32_t num_owning_cols>
inline uint32_t ring_incoming(uint32_t my_col) {
    uint32_t n = 0;
    for (uint32_t j = 1; j < k_cols; ++j) {
        n += nslice_owner<k_cols>(my_col, j) < num_owning_cols ? 1 : 0;
    }
    return n;
}

}  // namespace conv2d_3x3
