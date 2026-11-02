// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm flat regime (ROW_MAJOR WIDTH / BLOCK_SHARDED, shard width | 16 datums) — CB indices shared by the
// reader / compute / writer kernels. Must match the CB_* constants in flat_program_descriptor.py.
#pragma once

#include <cstdint>

namespace rms_norm_flat_cb {
constexpr uint32_t cb_x = 0;          // shard-backed input, tile-sized pages (full flat tiles)
constexpr uint32_t cb_xtail = 1;      // last partial flat tile of x (local copy, zero-filled)
constexpr uint32_t cb_gamma = 2;      // broadcast form: gamma_flat (member's w_s gamma datums repeated)
constexpr uint32_t cb_mms = 3;        // pass-1 0/1 constants: compressed S_0..S_{R-1}; broadcast B
constexpr uint32_t cb_sq = 4;         // x^2 flat tiles
constexpr uint32_t cb_partial = 5;    // per-unit partials (ring)
constexpr uint32_t cb_gathered = 6;   // owner: k partials per owned unit (slot j * k + m)
constexpr uint32_t cb_inv_owned = 7;  // owner: inv of the owned units
constexpr uint32_t cb_inv = 8;        // inv of every unit (slot u, written by unit u's owner)
constexpr uint32_t cb_out = 9;        // shard-backed output, tile-sized pages
constexpr uint32_t cb_outtail = 10;   // last partial flat tile of out (scratch)
constexpr uint32_t cb_mme = 11;       // compressed pass-2 expand matrices E_j (gamma folded in)
constexpr uint32_t cb_flags = 12;     // per-unit inv arrival flags (16 B each)
}  // namespace rms_norm_flat_cb
