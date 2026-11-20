// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tools/profiler/kernel_profiler.hpp"

// Opt-in per-stage zones (host env AGRMS_ZONES=1 -> define AGRMS_ZONES); compiled out otherwise.
#ifdef AGRMS_ZONES
#define AGZ(name) DeviceZoneScopedN(name)
#else
#define AGZ(name)
#endif

// CB indices — must match all_gather_rms_norm_program_descriptor.py.
namespace agrms_cb {
constexpr uint32_t cb_x = 0;  // landing slot + resident member block (FIRST CB: same address on every core / chip)
constexpr uint32_t cb_stage = 1;
constexpr uint32_t cb_gamma = 2;
constexpr uint32_t cb_beta = 3;
constexpr uint32_t cb_ones = 4;
constexpr uint32_t cb_mask = 5;
constexpr uint32_t cb_moment_a = 6;
constexpr uint32_t cb_moment_b = 7;
constexpr uint32_t cb_km = 8;
constexpr uint32_t cb_partial = 9;
constexpr uint32_t cb_gathered = 10;
constexpr uint32_t cb_rowstat = 11;
constexpr uint32_t cb_out = 12;
constexpr uint32_t cb_mstat = 13;
constexpr uint32_t cb_shift = 14;
constexpr uint32_t cb_kg = 15;
constexpr uint32_t cb_mu = 16;
constexpr uint32_t cb_affine_stage = 17;
constexpr uint32_t cb_norm = 18;
constexpr uint32_t cb_shift_mats = 19;
constexpr uint32_t cb_foreign = 20;
constexpr uint32_t cb_contrib = 21;
constexpr uint32_t cb_cent = 22;
constexpr uint32_t cb_own_ret = 23;  // owner combine: combined statistics of the owned rows (compute -> writer)
constexpr uint32_t cb_own_kg = 24;   // ln owner combine: K_g of the owned rows (bf16, compute -> writer)
constexpr uint32_t cb_dev = 25;      // ln owner combine scratch (row form)
constexpr uint32_t cb_wcol = 26;     // ln owner combine: member weights n_m (column 0, one row per member)
constexpr uint32_t cb_xg = 27;       // rms, AGRMS_XG_FIRST: x * gamma of the member block (accumulator format)
constexpr uint32_t cb_port_hdr = 29;  // port sender: packet-header ring (raw L1, sender only)
constexpr uint32_t cb_own = 28;      // ln, AGRMS_LN_OWN_PIVOT: copy of the own slice (input format, reader -> compute)
}  // namespace agrms_cb

// Owner combine (k > 8): tile-row r of a block is owned by member r % k; member m owns rows m, m + k, ...
namespace agrms_owner {
constexpr uint32_t ROW_GATHER = 32;  // members per gather tile (row m % 32 of gather tile m / 32)
inline uint32_t owned_rows(uint32_t rows, uint32_t member, uint32_t k) {
    return member < rows ? (rows - member + k - 1) / k : 0;
}
}  // namespace agrms_owner
