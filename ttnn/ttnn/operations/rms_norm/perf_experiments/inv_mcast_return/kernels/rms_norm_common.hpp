// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm — CB indices shared by the reader / compute / writer kernels.
// Must match the CB_* constants in rms_norm_program_descriptor.py.
#pragma once

#include <cstdint>

namespace rms_norm_cb {
constexpr uint32_t cb_x_rm = 0;            // RM: raw stick pieces (tile-sized pages) -> tilize
constexpr uint32_t cb_x = 1;               // resident x block (ping)
constexpr uint32_t cb_gamma = 2;           // expanded gamma slice (every row of tile j = gamma[32j..32j+31])
constexpr uint32_t cb_sumsq_tiles = 3;     // per tile-row DEST-accumulated sum of x^2
constexpr uint32_t cb_partial_sumsq = 4;   // k > 1: this member's per-row partial (column 0)
constexpr uint32_t cb_gathered_sumsq = 5;  // k > 1: k partials per tile-row (tile r*k + m)
constexpr uint32_t cb_inv_rms = 6;         // per tile-row rsqrt(mean + eps) (column 0)
constexpr uint32_t cb_out = 7;             // normalized tiles
constexpr uint32_t cb_out_rm = 8;          // RM: untilized stick pieces
constexpr uint32_t cb_x_alt = 9;           // resident x block (pong), x_block_depth == 2 only
constexpr uint32_t cb_sumsq_acc = 10;      // sharded TILE W tail: raw sum x^2 of the full W tiles (Accumulate CB)
constexpr uint32_t cb_tail_mask = 11;      // sharded TILE W tail: 0/1 row mask of the tail tile's valid columns
constexpr uint32_t cb_inv_owned = 12;      // owner combine (k > K_SYM_MAX_*): inv_rms of the rows this member owns
constexpr uint32_t cb_inv_flags = 13;      // owner combine, multicast flag return: k owner slots + 1 staging (16 B)
}  // namespace rms_norm_cb
