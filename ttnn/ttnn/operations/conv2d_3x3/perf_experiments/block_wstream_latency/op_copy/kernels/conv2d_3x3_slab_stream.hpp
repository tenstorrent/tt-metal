// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Streamed slab-direct fill (the descriptor's SD_STREAM_CHUNK_PX, kernels conv2d_3x3_slab_direct.hpp): compute
// multiplies each tile-row as soon as the slab pixels it reads have landed, instead of after the whole slab.
//
// A block's fill is a few pieces of input pixels in increasing flat order (Geom::fill_pieces), and the slab offset
// grows with the flat index, so "every piece pixel below slab pixel S has landed" is one monotone count: the slab CB's
// pushed pages. Each data-movement RISC streams its pieces (the reader: own and lower input shards, flat < B; the
// writer: the higher shards, flat >= B) in chunks of chunk_px pixels, chunk c under transaction id trid0 + c % ntrid
// with at most ntrid chunks in flight, and retires the landed chunks in issue order: `landed` is the slab pixel up to
// which its pieces landed (a chunk that ends an image row also covers the next slot's leading gap, zeroed at kernel
// start). The writer publishes its count in a semaphore; the reader, the slab CB's one producer, pushes the combined
// count (its own until its part is done, then the writer's). Compute waits, per subblock row, for the slab pages its
// in0 runs read (Geom::need_px). The descriptor streams only blocks of at least SD_STREAM_MIN_SUBBLOCK_ROWS subblock
// rows (a one- or two-row block needs nearly the whole slab for its first matmul call).
//
// Measured (Blackhole p150b, stem.conv1 512x1024x8 -> 64 s2, LoFi, bf8_b weights, bias + ReLU, unzoned, medians of 3
// alternating pairs): 11076 ns -> 10196 ns with the early round 0 (11036 ns with the early round 0 alone, 11056 ns with
// the stream alone, one run each). The critical cores (the last grid row, whose fill comes almost all from higher
// shards through the writer) start multiplying ~0.9 us earlier; the weight round (~3.5 us) is the bound now.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace conv2d_3x3::slab_stream {

// Up to max_pieces pieces of one RISC's part of a block's fill, issued in chunks. No division in the issue / retire
// path (the data-movement RISCs divide in software): chunk c's transaction-id slot advances by one and wraps.
template <uint32_t max_pieces, uint32_t ntrid, uint32_t trid0, uint32_t chunk_px, uint32_t in_w, uint32_t gap>
struct Stream {
    static_assert(ntrid >= 1 && trid0 >= 1 && trid0 + ntrid - 1 <= NOC_MAX_TRANSACTION_ID);
    uint32_t lo[max_pieces];
    uint32_t hi[max_pieces];
    uint32_t px[max_pieces];      // slab pixel of lo
    uint32_t hi_gap[max_pieces];  // gap if hi ends an image row (the next slot's zeroed leading gap), else 0
    uint32_t n = 0;
    uint32_t cur = 0;           // next piece to issue from
    uint32_t pos = 0;           // its next flat pixel
    uint32_t chunk_end[ntrid];  // landed count once the chunk in that slot retires
    uint32_t in_flight = 0;     // chunks issued and not retired
    uint32_t head_slot = 0;     // the oldest one's slot
    uint32_t tail_slot = 0;     // the next one's slot
    uint32_t landed = 0;        // slab pixels landed
    uint32_t region_end = 0;    // the count once every piece landed

    // A new block: no pieces, `region_end` = this part's end in slab pixels.
    FORCE_INLINE void reset(uint32_t end) {
        n = cur = pos = in_flight = head_slot = tail_slot = landed = 0;
        region_end = end;
    }

    FORCE_INLINE void add(uint32_t piece_lo, uint32_t piece_hi, uint32_t slab_px) {
        if (piece_lo >= piece_hi) {
            return;
        }
        ASSERT(n < max_pieces);
        if (n == 0) {
            pos = piece_lo;
        }
        lo[n] = piece_lo;
        hi[n] = piece_hi;
        px[n] = slab_px;
        hi_gap[n] = piece_hi % in_w == 0 ? gap : 0;
        ++n;
    }

    // No piece left to issue.
    FORCE_INLINE bool issued() const { return cur >= n; }
    FORCE_INLINE bool done() const { return cur >= n && in_flight == 0; }

    // Issue the next chunk if a transaction id is free: read(trid, flat_lo, flat_hi, slab_px_of_flat_lo) issues its
    // reads under trid.
    template <typename ReadFn>
    FORCE_INLINE bool issue_one(ReadFn&& read) {
        if (cur >= n || in_flight >= ntrid) {
            return false;
        }
        const uint32_t h = hi[cur];
        const uint32_t e = h - pos > chunk_px ? pos + chunk_px : h;
        read(trid0 + tail_slot, pos, e, px[cur] + (pos - lo[cur]));
        chunk_end[tail_slot] = px[cur] + (e - lo[cur]) + (e == h ? hi_gap[cur] : 0);
        tail_slot = tail_slot + 1 == ntrid ? 0 : tail_slot + 1;
        ++in_flight;
        pos = e;
        if (pos >= h) {
            ++cur;
            if (cur < n) {
                pos = lo[cur];
            }
        }
        return true;
    }

    // Retire the landed chunks in issue order; true if `landed` moved.
    FORCE_INLINE bool poll(uint32_t noc_id) {
        bool moved = false;
        while (in_flight != 0 && ncrisc_noc_read_with_transaction_id_flushed(noc_id, trid0 + head_slot)) {
            landed = chunk_end[head_slot];
            head_slot = head_slot + 1 == ntrid ? 0 : head_slot + 1;
            --in_flight;
            moved = true;
        }
        if (cur >= n && in_flight == 0 && landed != region_end) {
            landed = region_end;
            moved = true;
        }
        return moved;
    }
};

}  // namespace conv2d_3x3::slab_stream
