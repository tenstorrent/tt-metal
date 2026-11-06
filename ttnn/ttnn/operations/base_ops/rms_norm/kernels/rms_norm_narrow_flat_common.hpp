// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm narrow-flat regime (ROW_MAJOR WIDTH / BLOCK_SHARDED, shard width | 16 datums) — CB indices shared by the
// reader / compute / writer kernels. Must match the CB_* constants in rms_norm_narrow_flat_program_descriptor.py.
#pragma once

#include <cstdint>

namespace rms_norm_narrow_flat_cb {
constexpr uint32_t cb_x = 0;          // shard-backed input, tile-sized pages (full flat tiles)
constexpr uint32_t cb_xtail = 1;      // last partial flat tile of x (local copy, zero-filled)
constexpr uint32_t cb_mms = 3;        // pass-1 0/1 matrices S_0..S_{NB-1}
constexpr uint32_t cb_sq = 4;         // x^2 flat tiles
constexpr uint32_t cb_partial = 5;    // per-unit compressed partials (ring)
constexpr uint32_t cb_gathered = 6;   // owner: KT tiles per owned sub-unit (member m: tile m / Q, sub-block m % Q)
constexpr uint32_t cb_inv_owned = 7;  // owner: inv of the owned sub-units (sub-block 0)
constexpr uint32_t cb_inv = 8;        // inv of every unit (tile u, multicast by the unit's leader)
constexpr uint32_t cb_out = 9;        // shard-backed output, tile-sized pages
constexpr uint32_t cb_outtail = 10;   // last partial flat tile of out (scratch)
constexpr uint32_t cb_mme = 11;       // pass-2 expand matrices E_0..E_{NB-1} (gamma folded in)
constexpr uint32_t cb_flags =
    12;  // unit flags (at u * Q) + leaders collect counters + multicast flag source (16 B each)
}  // namespace rms_norm_narrow_flat_cb
