// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_to_all_matmul port packet walk, shared by the port reader (fills the packets) and the port sender (sends them),
// so both cut the same packets and batches.
//
// A port's entries are walked in runs ("groups") of consecutive entries to the same destination chip. Within a group:
//   pack = false  every entry is cut into ceil(slot_tiles / packet_tiles) packets (its last one may be short).
//   pack = true   (TRANSPORT, Refinement 4) the group is one stream of tiles cut into full packets; a packet that
//                 straddles an entry boundary carries n0 tiles of entry e and n1 of entry e + 1 (a 2-chunk fused
//                 scatter write + increment of entry e's arrival counter). Requires packet_tiles <= slot_tiles, so a
//                 packet straddles at most one boundary. Only the group's last packet may be short.
// A batch is up to batch_chunks packets, never crosses a group, and ends at a multiple of batch_chunks packets in the
// 2 x batch_chunks CB (a reservation never wraps).

#pragma once

#include <cstdint>

namespace a2amm_stream {

template <uint32_t slot_tiles, uint32_t packet_tiles, bool pack>
struct Walk {
    static_assert(!pack || packet_tiles <= slot_tiles, "packing needs packet_tiles <= slot_tiles");
    static constexpr uint32_t entry_packets = (slot_tiles + packet_tiles - 1) / packet_tiles;

    static constexpr uint32_t group_packets(uint32_t n_entries) {
        return pack ? (n_entries * slot_tiles + packet_tiles - 1) / packet_tiles : n_entries * entry_packets;
    }

    uint32_t e = 0;    // entry within the group
    uint32_t off = 0;  // tile offset within entry e

    // Tiles of the next packet: n0 in entry e from `off`, n1 in entry e + 1 from 0.
    void next(uint32_t n_entries, uint32_t& n0, uint32_t& n1) const {
        const uint32_t left_entry = slot_tiles - off;
        uint32_t n = packet_tiles;
        if constexpr (pack) {
            const uint32_t left_group = (n_entries - e) * slot_tiles - off;
            n = left_group < n ? left_group : n;
        } else {
            n = left_entry < n ? left_entry : n;
        }
        n0 = left_entry < n ? left_entry : n;
        n1 = n - n0;
    }
    void advance(uint32_t n) {
        off += n;
        if (off >= slot_tiles) {
            off -= slot_tiles;
            ++e;
        }
    }
};

}  // namespace a2amm_stream
