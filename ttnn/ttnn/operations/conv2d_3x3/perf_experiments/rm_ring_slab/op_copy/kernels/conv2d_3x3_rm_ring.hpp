// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// ROW_MAJOR ring slab (perf experiment rm_ring_slab; the descriptor's rm_ring_capable / rm_ring_px).
//
// cb_halo_slab is a FIFO ring of ring_px consecutive input pixels across a core's M-blocks: pixel v lives at ring slot
// (v - origin) mod ring_px, origin being the core's lowest window pixel. The fill is one monotone stream of flat pixels
// [origin, ...), each read once per core (the per-M-block slab reads each pixel of three disjoint kh bands, and every
// later block re-reads them: ~3x on the FLUX.2 shapes). Three cursors bound it:
//   landed   pixels below it are in L1 (a gather reads only those);
//   allowed  pixels below it may be written: the slot they reuse belongs to pixel v - ring_px, which no K-block still
//            ahead of compute reads. The reader learns compute's progress from the tap-direct entry FIFO (one K-block
//            deep): once K-block k's entries are all reserved, compute popped every entry of K-blocks < k;
//   cap      the next pixel a gather waits for. The opportunistic fill never issues past it before that wait, so the
//            wait (one transaction id for the whole stream) does not also wait for reads issued past it.
// The fill of pixel range [lo, hi) into ring slots is fill_range(): one TensorAccessor page read per pixel today; a
// bank-coalesced read can replace the per-range reader (read_px_range) without touching the cursors.
//
// Ring capacity (host mirror rm_ring_plan): max over M-blocks of (highest window end so far - lowest window start of
// this and every later block), so a need never waits on a slot it would itself free (no deadlock), and a block's
// pixels are at most one wrap apart (the tap-direct gather reduces ring positions once).

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "conv2d_3x3_slab_dataflow.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"

namespace conv2d_3x3 {

// In-image pixel extent of one M-block's three bands: band kh is [band_lo[kh], band_hi[kh]) (empty: band_hi == 0),
// the window [lo, hi) their union's hull.
struct RmWindow {
    int32_t lo;
    int32_t hi;
    int32_t band_lo[3];
    int32_t band_hi[3];
};

template <int32_t input_width, int32_t num_in_px, int32_t dilation>
FORCE_INLINE RmWindow rm_window(const SlabFootprint& fp) {
    RmWindow w{0x7FFFFFFF, 0, {0, 0, 0}, {0, 0, 0}};
    for (int32_t kh = 0; kh < 3; ++kh) {
        const int32_t v_lo = fp.f + kh * dilation * input_width;
        const int32_t lo = v_lo > 0 ? v_lo : 0;
        const int32_t hi = v_lo + fp.band_len < num_in_px ? v_lo + fp.band_len : num_in_px;
        if (lo >= hi) {
            continue;
        }
        w.band_lo[kh] = lo;
        w.band_hi[kh] = hi;
        w.lo = lo < w.lo ? lo : w.lo;
        w.hi = hi > w.hi ? hi : w.hi;
    }
    return w;
}

template <
    uint32_t output_width,
    int32_t input_width,
    int32_t stride,
    int32_t pad_top,
    int32_t pad_left,
    int32_t dilation,
    int32_t band_last_tap,
    int32_t num_in_px,
    uint32_t num_out_px,
    uint32_t block_m_tiles>
struct RmBlocks {
    static FORCE_INLINE RmWindow window(uint32_t m_tile_start, uint32_t actual_m_tiles, uint32_t m) {
        const uint32_t done = m * block_m_tiles;
        const uint32_t valid = actual_m_tiles - done < block_m_tiles ? actual_m_tiles - done : block_m_tiles;
        const uint32_t o_first = (m_tile_start + done) * 32;
        const uint32_t o_end_tiles = o_first + valid * 32;
        const SlabFootprint fp =
            block_footprint<output_width, input_width, stride, pad_top, pad_left, dilation, band_last_tap>(
                o_first, o_end_tiles < num_out_px ? o_end_tiles : num_out_px);
        return rm_window<input_width, num_in_px, dilation>(fp);
    }
    // Lowest window pixel of M-blocks [m_first, num_blocks) (0x7FFFFFFF when there is none). A block's window starts
    // at or above its first tap's pixel (clamped at 0), which never decreases from block to block.
    static FORCE_INLINE int32_t
    keep_lo(uint32_t m_tile_start, uint32_t actual_m_tiles, uint32_t m_first, uint32_t num_blocks) {
        int32_t lo = 0x7FFFFFFF;
        for (uint32_t m = m_first; m < num_blocks; ++m) {
            const uint32_t done = m * block_m_tiles;
            const uint32_t o_first = (m_tile_start + done) * 32;
            const int32_t ho = o_first / output_width;
            const int32_t wo = o_first % output_width;
            const int32_t f = (ho * stride - pad_top) * input_width + wo * stride - pad_left;
            if ((f > 0 ? f : 0) >= lo) {
                break;
            }
            const RmWindow w = window(m_tile_start, actual_m_tiles, m);
            lo = w.lo < lo ? w.lo : lo;
        }
        return lo;
    }
};

// min_free: transaction-id slots the fill leaves free (NIU outstanding-read counter of 255 per id); a chunk is issued
// only when chunk_px + min_free are free, and each chunk restarts the TaggedReads throttle count (chunk_px < 64), so
// up to 255 - min_free reads stay in flight: the fill is latency-bound per core (outstanding / round trip).
template <int32_t ring_px, uint32_t pixel_bytes, uint32_t trid, int32_t chunk_px, uint32_t min_free>
struct RmRing {
    static_assert(chunk_px >= 1 && chunk_px < static_cast<int32_t>(TRID_CHECK_EVERY_PACKETS), "one throttle window");
    static_assert(chunk_px + min_free <= NOC_MAX_TRANSACTION_ID_COUNT, "a chunk fits the id's counter");
    int32_t origin = 0;
    int32_t issued = 0;       // next pixel to read
    int32_t issued_slot = 0;  // its ring slot
    int32_t landed = 0;
    int32_t allowed = 0;
    int32_t cap = 0;
    uint32_t slab_l1 = 0;
    uint32_t reads_issued = 0;  // pixel reads issued (instrumentation)

    // The core's needed pixels: the union of its three kh bands (gaps between them, when the core spans less than d
    // image rows, are never read; their slots stay stale and nothing gathers them).
    int32_t seg_lo[3] = {0, 0, 0};
    int32_t seg_hi[3] = {0, 0, 0};
    uint32_t num_segs = 0;

    template <int32_t input_width, int32_t num_in_px, int32_t dilation>
    FORCE_INLINE void init(uint32_t l1, int32_t first_px, const SlabFootprint& core_fp) {
        slab_l1 = l1;
        origin = first_px;
        issued = landed = first_px;
        issued_slot = 0;
        allowed = first_px + ring_px;
        cap = first_px;
        num_segs = 0;
        for (int32_t kh = 0; kh < 3; ++kh) {
            const int32_t v_lo = core_fp.f + kh * dilation * input_width;
            const int32_t lo = v_lo > 0 ? v_lo : 0;
            const int32_t hi = v_lo + core_fp.band_len < num_in_px ? v_lo + core_fp.band_len : num_in_px;
            if (lo >= hi) {
                continue;
            }
            // Bands start in increasing order: merge into the last segment when they touch.
            if (num_segs > 0 && lo <= seg_hi[num_segs - 1]) {
                seg_hi[num_segs - 1] = hi > seg_hi[num_segs - 1] ? hi : seg_hi[num_segs - 1];
            } else {
                seg_lo[num_segs] = lo;
                seg_hi[num_segs] = hi;
                ++num_segs;
            }
        }
    }

    // Base the tap-direct gather adds to a pixel of a block whose lowest pixel is lo: v + base is v's ring slot or that
    // plus ring_px (one wrap), for every v in [lo, lo + ring_px).
    FORCE_INLINE int32_t base_for(int32_t lo) const {
        const int32_t q = (lo - origin) / ring_px;
        return -origin - q * ring_px;
    }

    FORCE_INLINE void release(int32_t live_lo) {
        const int32_t a = live_lo + ring_px;
        allowed = a > allowed ? a : allowed;
    }

    // Pixels [issued, hi) into their ring slots (split at the ring end) through read_px_range(v_lo, v_hi, l1).
    template <typename ReadRange>
    FORCE_INLINE void fill_range(int32_t hi, ReadRange&& read_px_range) {
        while (issued < hi) {
            const int32_t to_end = ring_px - issued_slot;
            const int32_t n = hi - issued < to_end ? hi - issued : to_end;
            for (uint32_t s = 0; s < num_segs; ++s) {
                const int32_t a = issued > seg_lo[s] ? issued : seg_lo[s];
                const int32_t z = issued + n < seg_hi[s] ? issued + n : seg_hi[s];
                if (a < z) {
                    read_px_range(a, z, slab_l1 + static_cast<uint32_t>(issued_slot + (a - issued)) * pixel_bytes);
                    reads_issued += z - a;
                }
            }
            issued += n;
            issued_slot += n;
            if (issued_slot == ring_px) {
                issued_slot = 0;
            }
        }
    }

    // Opportunistic: at most one chunk, only while the transaction id has room (never blocks on the NIU counter).
    // Returns true if it issued anything. The caller restores its own read transaction id afterwards.
    template <typename Reads, typename ReadRange>
    FORCE_INLINE bool step(Reads& reads, ReadRange&& read_px_range) {
        const int32_t lim = cap < allowed ? cap : allowed;
        if (issued >= lim) {
            return false;
        }
        if (noc_available_transactions(reads.noc.get_noc_id(), trid) < min_free + chunk_px) {
            return false;
        }
        reads.begin(trid);
        const int32_t hi = issued + chunk_px < lim ? issued + chunk_px : lim;
        fill_range(hi, read_px_range);
        return true;
    }

    // Blocking issue of [issued, hi) in chunks under the same in-flight bound.
    template <typename Reads, typename ReadRange>
    FORCE_INLINE void issue_to(Reads& reads, int32_t hi, ReadRange&& read_px_range) {
        while (issued < hi) {
            while (noc_available_transactions(reads.noc.get_noc_id(), trid) < min_free + chunk_px) {
            }
            reads.begin(trid);
            fill_range(issued + chunk_px < hi ? issued + chunk_px : hi, read_px_range);
        }
    }

    // Blocking: every pixel below x is issued and landed. x must be <= allowed (the plan's ring capacity guarantees
    // it).
    template <typename Reads, typename ReadRange>
    FORCE_INLINE void need(Reads& reads, int32_t x, ReadRange&& read_px_range) {
        if (x <= landed) {
            return;
        }
        ASSERT(x <= allowed);
        issue_to(reads, x, read_px_range);
        reads.wait(trid);
        landed = issued;
    }
};

}  // namespace conv2d_3x3
