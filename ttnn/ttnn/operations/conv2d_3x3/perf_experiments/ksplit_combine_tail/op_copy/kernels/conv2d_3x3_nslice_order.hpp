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

// ksplit_combine_tail experiment: COMBINE_TAIL bit 8 (far first) walks the remote owners farthest first (the reverse
// of the order above), so the expensive long-haul slice sends overlap the rest of the walk and the slices sent last
// are the one-hop ones; each slice keeps its RISC / NoC (+d on the reader's NoC0, -d on the writer's NoC1).
#ifndef COMBINE_TAIL
#define COMBINE_TAIL 0
#endif

namespace conv2d_3x3 {

constexpr bool nslice_far_first = (COMBINE_TAIL & 8) != 0;
// COMBINE_TAIL bit 16 (near last): the +-1 pair moves to the end of the remote walk (just before the own slice), the
// rest keep their ascending order: +2, -2, ..., then +1 / -1 (whichever keeps the RISCs alternating). A send's time
// grows with d (every core of the row sends at once, so each link carries d flows), and the last remote sends are the
// ones the owners wait for.
constexpr bool nslice_near_last = (COMBINE_TAIL & 16) != 0;
// COMBINE_TAIL bit 32 (reader far first): only the reader's +d sends run farthest first (+k/2 ... +1); the writer keeps
// -1, -2, ... and the own slice last, so its blocking sends stay short while it still streams weights. The reader has
// two slice periods per send, and its last send (+1) is one hop.
constexpr bool nslice_reader_far_first = (COMBINE_TAIL & 32) != 0;
// COMBINE_TAIL bit 128 (reader far middle): the reader sends +1 first (short, while the row's credits settle), then the
// rest farthest first: +1, +k/2, ..., +2. The writer keeps its order.
constexpr bool nslice_reader_far_mid = (COMBINE_TAIL & 128) != 0;

// Walk step j (< k_cols - 1) -> step of the base order (+1, -1, +2, -2, ...).
template <uint32_t k_cols>
inline uint32_t nslice_base_step(uint32_t j) {
    if constexpr (nslice_far_first) {
        return k_cols - 2 - j;
    } else if constexpr (nslice_near_last && k_cols >= 4) {
        return j + 2 <= k_cols - 2 ? j + 2 : j % 2;
    } else if constexpr (nslice_reader_far_mid) {
        constexpr uint32_t reader_steps = k_cols / 2;
        return (j % 2 == 0 && j > 0) ? 2 * reader_steps - j : j;
    } else if constexpr (nslice_reader_far_first) {
        constexpr uint32_t reader_steps = k_cols / 2;  // +1 .. +(k_cols / 2), the last one only for even k_cols
        return j % 2 == 0 ? 2 * reader_steps - 2 - j : j;
    } else {
        return j;
    }
}

template <uint32_t k_cols>
inline uint32_t nslice_owner(uint32_t my_col, uint32_t j) {
    if (j == k_cols - 1) {
        return my_col;
    }
    j = nslice_base_step<k_cols>(j);
    const uint32_t d = j / 2 + 1;
    return (j % 2 == 0) ? (my_col + d) % k_cols : (my_col + k_cols - d) % k_cols;
}

template <uint32_t k_cols>
inline bool nslice_reader_sends(uint32_t j) {
    if (j == k_cols - 1) {
        return false;
    }
    return nslice_base_step<k_cols>(j) % 2 == 0;
}

}  // namespace conv2d_3x3
