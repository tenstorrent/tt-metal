// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Compact faces-only W layout, shared by the W senders / receivers (push.hpp) and the matmul (compute).
//
// Only faces 0 and 2 of a W tile (columns 0..15; the 8 experts) are stored, in 1 KB face units: tile pair (2p, 2p+1)
// of a slot at 4p and 4p+1, so each tile's faces 0 / 2 sit at its base and base + 2 and its faces 1 / 3 (output
// columns 16..31, never read) alias real W data. A slot is one slice of ht tiles = 2*ht KB. Slots are NoC-major:
// slice c goes out on channel c % slice_nocs, and a channel's slices are consecutive slots (one multicast a NoC).

#pragma once

#include <cstdint>

namespace route_dispatch {

constexpr uint32_t kWFaceBytes = 1024;  // fp32 16x16 face

// Slices carried by NoC channel `channel`.
constexpr uint32_t w_channel_slices(uint32_t channel, uint32_t G, uint32_t slice_nocs) {
    return (G - channel + slice_nocs - 1) / slice_nocs;
}

// Compact slot of slice c.
constexpr uint32_t w_slot(uint32_t c, uint32_t G, uint32_t slice_nocs) {
    uint32_t slot = c / slice_nocs;
    for (uint32_t channel = 0; channel < c % slice_nocs; ++channel) {
        slot += w_channel_slices(channel, G, slice_nocs);
    }
    return slot;
}

// 1 KB-unit index (cb_w_faces page) of K tile kt of the slice at `slot`.
constexpr uint32_t w_face_tile(uint32_t slot, uint32_t ht, uint32_t kt) {
    return slot * 2 * ht + 4 * (kt / 2) + kt % 2;
}

}  // namespace route_dispatch
