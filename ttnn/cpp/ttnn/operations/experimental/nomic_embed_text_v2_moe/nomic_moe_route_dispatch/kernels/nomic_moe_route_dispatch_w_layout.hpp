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

// Split softmax (Perf R3.1): a core's tokens are cut into halves of 16 tokens (half h = face h % 2 (0 or 2) of score
// tile h / 2). The reader RISC selects halves [0, rh), the writer [rh, hn). The halves are ranked in the order the two
// RISCs need them (r0, w0, r1, w1, ..., then the longer side's rest); the half at rank pos is computed by member
// pos % G as its pass pos / G (the G members produce them while the RISCs consume them).
struct HalfSplit {
    uint32_t rh;  // reader halves
    uint32_t hn;  // halves with real tokens
};

constexpr HalfSplit half_split(uint32_t reader_tokens, uint32_t core_tokens) {
    const uint32_t hn = (core_tokens + 15) / 16;
    return {reader_tokens >= core_tokens ? hn : reader_tokens / 16, hn};
}

constexpr uint32_t half_rank(uint32_t half, HalfSplit split) {
    const uint32_t wh = split.hn - split.rh;
    const uint32_t pairs = split.rh < wh ? split.rh : wh;
    const uint32_t idx = half < split.rh ? half : half - split.rh;
    return idx < pairs ? 2 * idx + (half < split.rh ? 0 : 1) : 2 * pairs + (idx - pairs);
}

constexpr uint32_t half_at_rank(uint32_t pos, HalfSplit split) {
    const uint32_t wh = split.hn - split.rh;
    const uint32_t pairs = split.rh < wh ? split.rh : wh;
    if (pos < 2 * pairs) {
        return pos % 2 == 0 ? pos / 2 : split.rh + pos / 2;
    }
    return split.rh > wh ? pos - pairs : split.rh + (pos - pairs);
}

// Passes of member c.
constexpr uint32_t member_passes(uint32_t c, uint32_t G, HalfSplit split) {
    return split.hn > c ? (split.hn - c + G - 1) / G : 0;
}

// The router matmul packs one partial tile at a time, in the order the split softmax first needs them (by the lowest
// half rank in the tile; G = 1: tile order), then the tiles without real tokens. The exchange and the softmax follow
// the same order. order[i]: the i-th tile; slot[t]: the position of tile t.
template <uint32_t G, uint32_t rps>
struct TileOrder {
    uint32_t order[rps];
    uint32_t slot[rps];
    explicit TileOrder(HalfSplit split) {
        uint32_t count = 0;
        for (uint32_t t = 0; t < rps; ++t) {
            slot[t] = rps;
        }
        if constexpr (G > 1) {
            for (uint32_t pos = 0; pos < split.hn; ++pos) {
                const uint32_t t = half_at_rank(pos, split) / 2;
                if (slot[t] == rps) {
                    slot[t] = count;
                    order[count++] = t;
                }
            }
        }
        for (uint32_t t = 0; t < rps; ++t) {
            if (slot[t] == rps) {
                slot[t] = count;
                order[count++] = t;
            }
        }
    }
};

}  // namespace route_dispatch
