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

template <uint32_t k_cols>
inline uint32_t nslice_owner(uint32_t my_col, uint32_t j) {
    if (j == k_cols - 1) {
        return my_col;
    }
    const uint32_t d = j / 2 + 1;
    return (j % 2 == 0) ? (my_col + d) % k_cols : (my_col + k_cols - d) % k_cols;
}

template <uint32_t k_cols>
inline bool nslice_reader_sends(uint32_t j) {
    return j != k_cols - 1 && j % 2 == 0;
}

// nslice_paired_slices experiment: the weight-round schedule of the pair walk over steps [0, n) of the filtered order.
// The last `tail` steps are single-slice rounds (the walk's tail is weight-stream gated: a pair there would leave two
// slices of work after the last round lands); the rest go in pairs, an odd one out single. Fills first[r] / count[r]
// and returns the number of rounds. Compute and writer derive the same schedule.
inline uint32_t nslice_pair_rounds(uint32_t n, uint32_t tail, uint32_t* first, uint32_t* count) {
    const uint32_t paired_end = n > tail ? n - tail : 0;
    uint32_t r = 0;
    for (uint32_t s = 0; s < n; ++r) {
        const uint32_t c = (s + 1 < paired_end) ? 2 : 1;
        first[r] = s;
        count[r] = c;
        s += c;
    }
    return r;
}

}  // namespace conv2d_3x3
