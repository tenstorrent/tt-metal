// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The head reduction's publish schedule, which its compute and dataflow kernels (and, for the row slot, the host's
// output buffer) must follow identically. Both kernels have already returned for groups at or beyond `active`. The last
// active group composes the whole head and publishes it once: unchanged when it is the only active group, otherwise in
// its last composition stage, the one at which every later distance exceeds its index.
namespace kda_reduce {

inline bool publishes_unchanged(uint32_t active) { return active == 1; }
inline bool publishes_at_stage(uint32_t group, uint32_t active, uint32_t distance) {
    return group + 1 == active && 2 * distance >= active;
}

// Each published tile row (Kt tiles of E or Vt tiles of B) occupies one slot of this many output-buffer entries; the
// buffer holds two slots. Tiles are packed and read at offsets from the row's start, but on WH/BH the ring wraps only
// at a push, so every reservation must begin on a slot boundary. Rows of Kt and Vt entries would straddle the ring's
// end whenever Kt and Vt do not divide its capacity (Kt = 3, Vt = 4: the third E row starts at entry 6 of 8).
constexpr uint32_t published_row_entries(uint32_t Kt, uint32_t Vt) { return Kt > Vt ? Kt : Vt; }

}  // namespace kda_reduce
