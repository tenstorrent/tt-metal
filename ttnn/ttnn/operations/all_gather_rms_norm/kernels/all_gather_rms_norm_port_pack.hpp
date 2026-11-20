// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Packetization of one outgoing (wave, shard) entry — TRANSPORT, shared by the port reader and the port sender so
// both walk exactly the same packets (one cb_port_stage slot per packet).
//
// An entry is a list of runs [(core xy, tile offset in cb_x, tiles)] given as runtime args (xy, off | n << 16).
// A packet carries at most `chunk_pages` tiles (the fabric's max payload). With `pack` (fabric scatter write) a
// packet takes tiles from up to MAX_PIECES consecutive runs (different cores / non-contiguous offsets), so a member
// run that is not a multiple of chunk_pages no longer leaves a short packet behind (focus case: 12-tile runs, 7 + 5
// -> full 7-tile packets). Packets never span two entries: an entry's arrival increment must not wait on the next
// entry's (relay) dependency. Without `pack` a packet is one piece of one run (the original packetization).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace agrms_pack {

constexpr uint32_t MAX_PIECES = 4;  // NOC_SCATTER_WRITE_MAX_CHUNKS

struct Piece {
    uint32_t xy;   // NoC x | y << 16 of the compute core (same on every chip: mirror layout)
    uint32_t off;  // tile offset in that core's cb_x
    uint32_t n;    // tiles
};

template <bool pack, uint32_t chunk_pages>
struct EntryPacker {
    uint32_t a;       // runtime-arg index of the entry's first run
    uint32_t n_runs;  // runs of the entry
    uint32_t r = 0;   // current run
    uint32_t t = 0;   // tiles of run r already packed

    bool done() const { return r >= n_runs; }

    // Next packet of the entry into p[]; returns the piece count, tiles in `tiles`, `last` = it ends the entry.
    uint32_t next(Piece* p, uint32_t& tiles, bool& last) {
        uint32_t np = 0;
        tiles = 0;
        while (r < n_runs && tiles < chunk_pages && np < (pack ? MAX_PIECES : 1u)) {
            const uint32_t xy = get_arg_val<uint32_t>(a + 2 * r);
            const uint32_t offn = get_arg_val<uint32_t>(a + 2 * r + 1);
            const uint32_t n = offn >> 16;
            const uint32_t take = (n - t) < (chunk_pages - tiles) ? (n - t) : (chunk_pages - tiles);
            p[np++] = Piece{xy, (offn & 0xFFFF) + t, take};
            tiles += take;
            t += take;
            if (t == n) {
                ++r;
                t = 0;
            }
        }
        last = r >= n_runs;
        return np;
    }
};

}  // namespace agrms_pack
