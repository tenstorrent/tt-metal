// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// ROW_MAJOR ring slab with the bank-run fill (the descriptor's rm_ring_capable / rm_ring_plan / RM_RING_*).
//
// cb_halo_slab is a ring of ring_px consecutive input pixels across a core's M-blocks: pixel v lives at ring slot
// (v - origin) mod ring_px, origin being the core's lowest window pixel. The fill is one monotone stream over the
// core's needed pixels (the union of its three kh bands), each read from DRAM once per core; the per-M-block slab
// re-reads the bands every later block covers again (~3x where bm * 32 < W).
//
// Bank-run reads: the stream is cut into chunks of chunk_px consecutive pixels [a, e). DRAM-interleaved pages
// p, p + NB, p + 2 NB, ... sit back to back in one bank, so a chunk is NB reads (one per bank) into a staging chunk
// (bank s's run at s * slot_bytes), then de-interleaved into the ring with one self-aimed NoC read per pixel (a run
// wraps at the ring end). nstg staging chunks: up to nstg chunks of DRAM reads stay in flight while the reader
// de-interleaves or gathers. The staging is its own CB (counted in the L1 solve), so the fill can run while compute
// tilizes from cb_im2col_rm.
//
// Cursors:
//   copied   every needed pixel below it has its de-interleave copy issued;
//   landed   ... and landed (a gather reads only those);
//   allowed  pixels below it may be written: the slot they reuse belongs to pixel v - ring_px, which no K-block still
//            ahead of compute reads. The reader learns compute's progress from the tap-direct entry FIFO (one K-block
//            deep): once K-block k - 1's entries are all reserved, compute popped every entry of K-blocks < k - 1.
// Chunks are cut at `allowed` (so every staged chunk can be copied) and, in a band wait, at the band's end. The host
// plan (rm_ring_plan) sizes the ring so a band wait never needs a pixel past `allowed` (it would wait on a slot only it
// could free: no deadlock), and a block's pixels are at most one wrap apart (the tap-direct gather reduces ring
// positions once). Measured (Blackhole p150b, bit-identical): 1026x2048 32->128 2.61 -> 1.88 ms, 1026x2048 128->3 5.23
// -> 2.47 ms, 514x1024 128->3 1.41 -> 0.81 ms (perf_experiments/rm_ring_bank_run).
//
// Raw NoC (capability): dataflow_kernel_lib::read_with_state fixes the source address; the de-interleave changes both
// addresses on every read (as the graduated bank-run fill).

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "conv2d_3x3_slab_dataflow.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"

namespace conv2d_3x3 {

constexpr int32_t RM_INF = 0x7FFFFFFF;

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
    RmWindow w{RM_INF, 0, {0, 0, 0}, {0, 0, 0}};
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

// The lowest pixel of bands kh0.. of a window (RM_INF when they are all empty).
FORCE_INLINE int32_t rm_first_band_lo(const RmWindow& w, int32_t kh0) {
    for (int32_t b = kh0; b < 3; ++b) {
        if (w.band_hi[b] > 0) {
            return w.band_lo[b];
        }
    }
    return RM_INF;
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
    // Lowest window pixel of M-blocks [m_first, num_blocks) (RM_INF when there is none). A block's window starts at or
    // above its first tap's pixel (clamped at 0), which never decreases from block to block.
    static FORCE_INLINE int32_t
    keep_lo(uint32_t m_tile_start, uint32_t actual_m_tiles, uint32_t m_first, uint32_t num_blocks) {
        int32_t lo = RM_INF;
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

template <
    int32_t ring_px,
    uint32_t pixel_bytes,
    uint32_t nb,
    uint32_t nstg,
    int32_t chunk_px,
    uint32_t copy_trid,
    uint32_t stage_trid0>
struct RmBankRing {
    static_assert(nstg >= 1 && stage_trid0 + nstg - 1 <= NOC_MAX_TRANSACTION_ID, "staging trids");
    static_assert(chunk_px >= 1 && chunk_px <= ring_px, "a chunk fits the ring");
    static constexpr uint32_t slot_bytes = (static_cast<uint32_t>(chunk_px) + nb - 1) / nb * pixel_bytes;
    static constexpr uint32_t chunk_bytes = nb * slot_bytes;
    static constexpr uint32_t ring_bytes = static_cast<uint32_t>(ring_px) * pixel_bytes;

    uint32_t ring_l1 = 0;
    uint32_t stage_l1 = 0;
    int32_t origin = 0;
    // The core's needed pixels: the union of its three kh bands, segments [seg_lo, seg_hi) in increasing order (gaps
    // between them, when the core spans less than d image rows, are never read: their slots stay stale and nothing
    // gathers them).
    int32_t seg_lo[3] = {0, 0, 0};
    int32_t seg_hi[3] = {0, 0, 0};
    uint32_t num_segs = 0;
    // DRAM issue cursor: the next chunk's segment and first pixel.
    uint32_t iss_seg = 0;
    int32_t iss_a = 0;
    uint32_t n_issued = 0;
    uint32_t n_proc = 0;
    // Staged chunks: pixels [c_lo, c_hi), and the stream position once they are copied (the next segment's start
    // when the chunk ends its segment).
    int32_t c_lo[nstg];
    int32_t c_hi[nstg];
    int32_t c_after[nstg];
    int32_t copied = 0;
    int32_t landed = 0;
    int32_t allowed = 0;
    bool copies_out = false;

    template <int32_t input_width, int32_t num_in_px, int32_t dilation>
    FORCE_INLINE void init(uint32_t ring, uint32_t stage, int32_t first_px, const SlabFootprint& core_fp) {
        ring_l1 = ring;
        stage_l1 = stage;
        allowed = first_px + ring_px;
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
        // Slots count from the lowest pixel either holds (they coincide on every geometry the plan covers).
        origin = num_segs > 0 && seg_lo[0] < first_px ? seg_lo[0] : first_px;
        iss_seg = 0;
        iss_a = copied = landed = num_segs > 0 ? seg_lo[0] : RM_INF;
        n_issued = n_proc = 0;
    }

    // Base the tap-direct gather adds to a pixel of a block whose lowest pixel is lo: v + base is v's ring slot or that
    // plus ring_px (one wrap), for every v in [lo, lo + ring_px).
    FORCE_INLINE int32_t base_for(int32_t lo) const {
        const int32_t q = (lo - origin) / ring_px;
        return -origin - q * ring_px;
    }

    FORCE_INLINE void release(int32_t live_lo) {
        if (live_lo < RM_INF) {
            const int32_t a = live_lo + ring_px;
            allowed = a > allowed ? a : allowed;
        }
    }

    template <typename Reads>
    FORCE_INLINE void land_copies(Reads& reads) {
        if (copies_out) {
            reads.wait(copy_trid);
            copies_out = false;
        }
        landed = copied;
    }

    FORCE_INLINE bool can_issue() const { return n_issued - n_proc < nstg && iss_seg < num_segs && iss_a < allowed; }

    // The next chunk (at most to `to`, never past `allowed`, so every staged chunk can be copied): NB bank-run DRAM
    // reads into its staging slot, whose previous chunk's copies must have landed.
    template <typename Reads, typename Accessor>
    FORCE_INLINE void issue(Reads& reads, const Accessor& input, int32_t to) {
        const uint32_t slot = n_issued % nstg;
        if (n_issued >= nstg) {
            land_copies(reads);
        }
        const int32_t a = iss_a;
        const int32_t s_hi = seg_hi[iss_seg];
        int32_t e = a + chunk_px < s_hi ? a + chunk_px : s_hi;
        e = e < to ? e : to;
        e = e < allowed ? e : allowed;
        ASSERT(a < e);
        const uint32_t base = stage_l1 + slot * chunk_bytes;
        reads.begin(stage_trid0 + slot);
        for (int32_t s = 0; s < static_cast<int32_t>(nb) && a + s < e; ++s) {
            const uint32_t run = static_cast<uint32_t>(e - a - s + nb - 1) / nb;
            reads.read(input.get_noc_addr(a + s), base + s * slot_bytes, run * pixel_bytes);
        }
        c_lo[slot] = a;
        c_hi[slot] = e;
        iss_a = e;
        if (e == s_hi) {
            ++iss_seg;
            iss_a = iss_seg < num_segs ? seg_lo[iss_seg] : RM_INF;
        }
        c_after[slot] = iss_a;
        ++n_issued;
    }

    // The oldest staged chunk into its ring slots (blocking on its DRAM reads).
    template <typename Reads>
    FORCE_INLINE void process(Reads& reads) {
        const uint32_t slot = n_proc % nstg;
        const int32_t a = c_lo[slot];
        const int32_t e = c_hi[slot];
        ASSERT(e <= allowed);
        reads.wait(stage_trid0 + slot);
        const uint32_t base = stage_l1 + slot * chunk_bytes;
        const uint32_t noc = reads.noc.get_noc_id();
        reads.begin(copy_trid);
        ncrisc_noc_read_set_state<DM_DEDICATED_NOC, true>(noc, NCRISC_RD_CMD_BUF, get_noc_addr(base, noc), pixel_bytes);
        const int32_t slot_px = (a - origin) % ring_px;
        const uint32_t ring_end = ring_l1 + ring_bytes;
        for (int32_t s = 0; s < static_cast<int32_t>(nb) && a + s < e; ++s) {
            uint32_t src = base + s * slot_bytes;
            int32_t x = slot_px + s;
            if (x >= ring_px) {
                x -= ring_px;
            }
            uint32_t dst = ring_l1 + static_cast<uint32_t>(x) * pixel_bytes;
            for (int32_t q = a + s; q < e; q += nb) {
                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, true>(noc, NCRISC_RD_CMD_BUF, src, dst);
                reads.count_packets(1);
                src += pixel_bytes;
                dst += nb * pixel_bytes;
                if (dst >= ring_end) {
                    dst -= ring_bytes;
                }
            }
        }
        reads.drop_fixed_state();
        copies_out = true;
        copied = c_after[slot];
        ++n_proc;
    }

    // Opportunistic (compute is behind): one DRAM chunk issue, else one landed chunk's de-interleave. Returns true if
    // it did anything; the caller then restores its own read transaction id and command-buffer state.
    template <typename Reads, typename Accessor>
    FORCE_INLINE bool step(Reads& reads, const Accessor& input) {
        if (can_issue()) {
            issue(reads, input, RM_INF);
            return true;
        }
        if (n_proc < n_issued &&
            ncrisc_noc_read_with_transaction_id_flushed(reads.noc.get_noc_id(), stage_trid0 + n_proc % nstg)) {
            process(reads);
            return true;
        }
        return false;
    }

    // Blocking, whole chunks: every needed pixel below x is in the ring, the chunks that bring it in issued whole (up
    // to `allowed`), not cut at x. The kh-reuse stream waits for one unit's pixels at a time; cutting there would turn
    // a lagging fill into one DRAM round trip per unit.
    template <typename Reads, typename Accessor>
    FORCE_INLINE void need_whole(Reads& reads, const Accessor& input, int32_t x) {
        if (x <= landed) {
            return;
        }
        while (copied < x) {
            if (can_issue() && iss_a < x) {
                issue(reads, input, RM_INF);
            } else {
                process(reads);
            }
        }
        land_copies(reads);
    }

    // Blocking: every needed pixel below x is in the ring. x is within the plan's capacity of the live pixels
    // (x <= allowed), so the chunks it needs can be issued.
    template <typename Reads, typename Accessor>
    FORCE_INLINE void need(Reads& reads, const Accessor& input, int32_t x) {
        if (x <= landed) {
            return;
        }
        ASSERT(x <= allowed);
        while (copied < x) {
            if (can_issue() && iss_a < x) {
                issue(reads, input, x);
            } else {
                process(reads);
            }
        }
        land_copies(reads);
    }
};

}  // namespace conv2d_3x3
