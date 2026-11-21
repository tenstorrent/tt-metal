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
//
// Split startup fill (RM_SPLIT, `w_nstg` > 0): the first window (the stream's first segment up to the first `allowed`,
// ~2 image rows) must land before the core's first M-block can finish, on every core at once. The reader publishes
// it in L1 words (RmSplitWords) and the writer (BRISC, NoC1; RmRingWriterPart) fills part of it through its own
// staging chunks. Both number the window's chunks from its first pixel; the reader claims them from the front, the
// writer from the back, until they meet, so each core's split follows whichever RISC and NoC deliver faster there. A
// claim is a Dekker pair: each side publishes its claim count, then reads the other's. The writer marks a claim
// undecided until it has read the reader's count and gives the chunk up if the reader claimed it; the reader reads
// only decided writer claims; so every chunk has exactly one owner. The writer sets `done` once its chunks landed in
// the ring; the reader's `landed` passes the first pixel of the writer's part only then. The writer stages through
// the last w_nstg of the nstg staging chunks; after the window (the writer done, nothing staged) the reader fills alone
// through all nstg again: with one chunk it would wait for each chunk's copies before the next chunk's DRAM reads.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "conv2d_3x3_slab_dataflow.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"

namespace conv2d_3x3 {

constexpr int32_t RM_INF = 0x7FFFFFFF;

// Split startup fill handshake words (L1 semaphores of this core).
struct RmSplitWords {
    volatile tt_l1_ptr uint32_t* lo;       // the window's first pixel (reader)
    volatile tt_l1_ptr uint32_t* hi;       // its end pixel, published last (0: not yet)
    volatile tt_l1_ptr uint32_t* origin;   // the ring origin (reader)
    volatile tt_l1_ptr uint32_t* r_claim;  // chunks the reader claimed from the front
    volatile tt_l1_ptr uint32_t* w_claim;  // chunks the writer claimed from the back, | RM_DECIDED once checked
    volatile tt_l1_ptr uint32_t* w_done;   // 1 once the writer's chunks landed in the ring

    template <uint32_t lo_id, uint32_t hi_id, uint32_t origin_id, uint32_t rc_id, uint32_t wc_id, uint32_t done_id>
    static FORCE_INLINE RmSplitWords from_semaphores() {
        auto w = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
        return RmSplitWords{w(lo_id), w(hi_id), w(origin_id), w(rc_id), w(wc_id), w(done_id)};
    }
};
constexpr uint32_t RM_DECIDED = 0x80000000u;

// The other RISC's word.
FORCE_INLINE uint32_t rm_load(volatile tt_l1_ptr uint32_t* p) {
    invalidate_l1_cache();
    return *p;
}
// Visible to the other RISC before any later load or store of this one.
FORCE_INLINE void rm_store(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    *p = v;
    asm volatile("fence" ::: "memory");
}

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
    uint32_t stage_trid0,
    uint32_t w_nstg = 0>
struct RmBankRing {
    static constexpr bool split = w_nstg > 0;
    static_assert(w_nstg < nstg, "the reader keeps a staging chunk");
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
    // Split startup fill: the handshake words; the window [ep_lo, ep_hi) while it is still claimable (ep_open), its
    // chunks and the reader's front claims, the stream position after it (iss_seg / iss_a point there meanwhile); the
    // first pixel of the writer's part until the writer is seen done (RM_INF: none, or done).
    RmSplitWords sw{};
    bool ep_open = false;
    int32_t ep_lo = 0;
    int32_t ep_hi = 0;
    uint32_t ep_n = 0;
    uint32_t ep_rc = 0;
    int32_t ep_next = 0;
    int32_t wp_lo = RM_INF;
    // Staging chunks the reader cycles through: nstg - w_nstg during the split window, then nstg.
    uint32_t nstg_now = nstg;

    // Staging chunk of the n-th chunk (both counts compile-time: no runtime division).
    FORCE_INLINE uint32_t slot_of(uint32_t n) const {
        if constexpr (split) {
            if (nstg_now != nstg) {
                return n % (nstg - w_nstg);
            }
        }
        return n % nstg;
    }

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
        if constexpr (split) {
            if (num_segs > 0) {
                // The window: the first segment up to `allowed`.
                ep_lo = seg_lo[0];
                ep_hi = seg_hi[0] < allowed ? seg_hi[0] : allowed;
                ep_n = static_cast<uint32_t>(ep_hi - ep_lo + chunk_px - 1) / chunk_px;
                ep_rc = 0;
                ep_open = true;
                iss_a = ep_hi;
                if (ep_hi == seg_hi[0]) {
                    iss_seg = 1;
                    iss_a = num_segs > 1 ? seg_lo[1] : RM_INF;
                }
                ep_next = iss_a;
                // The last w_nstg staging chunks are the writer's until it is done; the window goes out last.
                nstg_now = nstg - w_nstg;
                rm_store(sw.origin, static_cast<uint32_t>(origin));
                rm_store(sw.lo, static_cast<uint32_t>(ep_lo));
                rm_store(sw.hi, static_cast<uint32_t>(ep_hi));
            }
        }
    }

    // Split startup fill: the writer's claims, once its latest one is decided.
    FORCE_INLINE uint32_t writer_claims() const {
        uint32_t w;
        do {
            w = rm_load(sw.w_claim);
        } while (w != 0 && (w & RM_DECIDED) == 0);
        return w & ~RM_DECIDED;
    }

    // The front and back claims met: the writer owns the window's last wc chunks. Every pixel from the reader's last
    // chunk to ep_next is then the writer's (or in a segment gap nothing reads).
    FORCE_INLINE void close_window(uint32_t wc) {
        ep_open = false;
        if (wc > 0) {
            wp_lo = ep_lo + static_cast<int32_t>(ep_n - wc) * chunk_px;
        }
        if (n_issued > n_proc) {
            c_after[slot_of(n_issued - 1)] = ep_next;
        } else {
            copied = ep_next;
        }
    }

    // After the split window: every staging chunk back to the reader once the writer is done with its own and nothing
    // is staged at a slot index both counts agree on.
    FORCE_INLINE void widen() {
        if constexpr (split) {
            if (nstg_now < nstg && !ep_open && wp_lo == RM_INF && n_issued == n_proc && n_issued % nstg == 0) {
                nstg_now = nstg;
            }
        }
    }

    // The window's next front chunk, or false (the claims met: the window is closed).
    FORCE_INLINE bool claim_front() {
        rm_store(sw.r_claim, ep_rc + 1);
        const uint32_t wc = writer_claims();
        if (ep_rc + wc < ep_n) {
            ++ep_rc;
            return true;
        }
        close_window(wc);
        return false;
    }

    // The DRAM reads of the front chunk just claimed.
    template <typename Reads, typename Accessor>
    FORCE_INLINE void issue_claimed(Reads& reads, const Accessor& input) {
        const uint32_t slot = slot_of(n_issued);
        if (n_issued >= nstg_now) {
            land_copies(reads);
        }
        const int32_t a = ep_lo + static_cast<int32_t>(ep_rc - 1) * chunk_px;
        const int32_t e = a + chunk_px < ep_hi ? a + chunk_px : ep_hi;
        const uint32_t base = stage_l1 + slot * chunk_bytes;
        reads.begin(stage_trid0 + slot);
        for (int32_t s = 0; s < static_cast<int32_t>(nb) && a + s < e; ++s) {
            const uint32_t run = static_cast<uint32_t>(e - a - s + nb - 1) / nb;
            reads.read(input.get_noc_addr(a + s), base + s * slot_bytes, run * pixel_bytes);
        }
        c_lo[slot] = a;
        c_hi[slot] = e;
        c_after[slot] = e;
        ++n_issued;
    }

    // Every needed pixel below it has landed, the writer's part included.
    FORCE_INLINE int32_t landed_all() {
        if constexpr (split) {
            if (wp_lo < RM_INF) {
                if (rm_load(sw.w_done) == 0) {
                    return landed < wp_lo ? landed : wp_lo;
                }
                wp_lo = RM_INF;
            }
        }
        return landed;
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

    // (Never while the split startup window is open: chunks are issued and copied in stream order.)
    FORCE_INLINE bool can_issue() const {
        return !ep_open && n_issued - n_proc < nstg_now && iss_seg < num_segs && iss_a < allowed;
    }

    // The next chunk (at most to `to`, never past `allowed`, so every staged chunk can be copied): NB bank-run DRAM
    // reads into its staging slot, whose previous chunk's copies must have landed.
    template <typename Reads, typename Accessor>
    FORCE_INLINE void issue(Reads& reads, const Accessor& input, int32_t to) {
        const uint32_t slot = slot_of(n_issued);
        if (n_issued >= nstg_now) {
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
        const uint32_t slot = slot_of(n_proc);
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
        if constexpr (split) {
            if (ep_open && n_issued - n_proc < nstg_now && claim_front()) {
                issue_claimed(reads, input);
                return true;
            }
            widen();
        }
        if (can_issue()) {
            issue(reads, input, RM_INF);
            return true;
        }
        if (n_proc < n_issued &&
            ncrisc_noc_read_with_transaction_id_flushed(reads.noc.get_noc_id(), stage_trid0 + slot_of(n_proc))) {
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

    // Blocking, as need(), with whole chunks after the split window (up to `allowed`, not cut at x): the row stream
    // waits for one tile-row's pixels at a time, and cutting there would cost one DRAM round trip per tile-row.
    template <typename Reads, typename Accessor>
    FORCE_INLINE void need_stream(Reads& reads, const Accessor& input, int32_t x) {
        if (x <= landed_all()) {
            return;
        }
        ASSERT(x <= allowed);
        if constexpr (split) {
            widen();
        }
        while (copied < x) {
            if (ep_open) {
                if (n_issued - n_proc < nstg_now && ep_lo + static_cast<int32_t>(ep_rc) * chunk_px < x) {
                    if (claim_front()) {
                        issue_claimed(reads, input);
                    }
                    continue;
                }
            } else if (can_issue() && iss_a < x) {
                issue(reads, input, RM_INF);
                continue;
            }
            process(reads);
        }
        land_copies(reads);
        if constexpr (split) {
            if (landed_all() < x) {
                MaybeDeviceZoneScope("reader_rm_split_wait");
                while (rm_load(sw.w_done) == 0) {
                }
                wp_lo = RM_INF;
            }
        }
    }

    // Blocking: every needed pixel below x is in the ring. x is within the plan's capacity of the live pixels
    // (x <= allowed), so the chunks it needs can be issued.
    template <typename Reads, typename Accessor>
    FORCE_INLINE void need(Reads& reads, const Accessor& input, int32_t x) {
        if (x <= landed_all()) {
            return;
        }
        ASSERT(x <= allowed);
        if constexpr (split) {
            widen();
        }
        while (copied < x) {
            if (ep_open) {
                // Split startup window: the reader's front chunks (a claim that finds the writer's closes it: the
                // stream then goes on past the window).
                if (n_issued - n_proc < nstg_now && ep_lo + static_cast<int32_t>(ep_rc) * chunk_px < x) {
                    if (claim_front()) {
                        issue_claimed(reads, input);
                    }
                    continue;
                }
            } else if (can_issue() && iss_a < x) {
                issue(reads, input, x);
                continue;
            }
            process(reads);
        }
        land_copies(reads);
        if constexpr (split) {
            if (landed_all() < x) {
                MaybeDeviceZoneScope("reader_rm_split_wait");
                while (rm_load(sw.w_done) == 0) {
                }
                wp_lo = RM_INF;
            }
        }
    }
};

// The split startup fill's writer part (BRISC, NoC1): claims the window's chunks from the back, one DRAM read per bank
// into its own nstg staging chunks, then one self-aimed read per pixel into the ring (as RmBankRing). The writer calls
// serve() while it waits for output tiles, so one call does at most one bank's de-interleave run (chunk_px / nb
// pixels): a whole chunk's copies would hold back the output store.
template <
    int32_t ring_px,
    uint32_t pixel_bytes,
    uint32_t nb,
    uint32_t nstg,
    int32_t chunk_px,
    uint32_t copy_trid,
    uint32_t stage_trid0>
struct RmRingWriterPart {
    static_assert(nstg >= 1 && stage_trid0 + nstg - 1 <= NOC_MAX_TRANSACTION_ID, "staging trids");
    static constexpr uint32_t slot_bytes = (static_cast<uint32_t>(chunk_px) + nb - 1) / nb * pixel_bytes;
    static constexpr uint32_t chunk_bytes = nb * slot_bytes;
    static constexpr uint32_t ring_bytes = static_cast<uint32_t>(ring_px) * pixel_bytes;

    uint32_t ring_l1 = 0;
    uint32_t stage_l1 = 0;
    RmSplitWords sw{};
    int32_t origin = 0;
    // The window, its chunks, this writer's back claims; started (the reader published it), met (no more claims),
    // done (every claimed chunk landed and `done` is published).
    int32_t lo = 0;
    int32_t hi = 0;
    uint32_t n = 0;
    uint32_t wc = 0;
    bool started = false;
    bool met = false;
    bool done = false;
    uint32_t n_issued = 0;
    uint32_t n_proc = 0;
    int32_t c_lo[nstg];
    int32_t c_hi[nstg];
    bool copies_out = false;
    // The oldest staged chunk's next bank run to de-interleave (0: not started).
    uint32_t proc_bank = 0;

    FORCE_INLINE void init(uint32_t ring, uint32_t stage, const RmSplitWords& words) {
        ring_l1 = ring;
        stage_l1 = stage;
        sw = words;
    }

    // Claim the next chunk from the back (false: the claims met).
    FORCE_INLINE bool claim_back(uint32_t& j) {
        if (wc >= n) {
            return false;
        }
        j = n - 1 - wc;
        if (rm_load(sw.r_claim) > j) {
            return false;
        }
        rm_store(sw.w_claim, wc + 1);
        if (rm_load(sw.r_claim) > j) {
            rm_store(sw.w_claim, wc | RM_DECIDED);
            return false;
        }
        ++wc;
        rm_store(sw.w_claim, wc | RM_DECIDED);
        return true;
    }

    template <typename Reads, typename Accessor>
    FORCE_INLINE void issue(Reads& reads, const Accessor& input, uint32_t j) {
        const uint32_t slot = n_issued % nstg;
        if (n_issued >= nstg && copies_out) {
            // The slot's previous chunk's copies read it.
            reads.wait(copy_trid);
            copies_out = false;
        }
        const int32_t a = lo + static_cast<int32_t>(j) * chunk_px;
        const int32_t e = a + chunk_px < hi ? a + chunk_px : hi;
        const uint32_t base = stage_l1 + slot * chunk_bytes;
        reads.begin(stage_trid0 + slot);
        for (int32_t s = 0; s < static_cast<int32_t>(nb) && a + s < e; ++s) {
            const uint32_t run = static_cast<uint32_t>(e - a - s + nb - 1) / nb;
            reads.read(input.get_noc_addr(a + s), base + s * slot_bytes, run * pixel_bytes);
        }
        c_lo[slot] = a;
        c_hi[slot] = e;
        ++n_issued;
    }

    // One bank run of the oldest staged chunk (its DRAM reads landed) into the ring.
    template <typename Reads>
    FORCE_INLINE void process_bank(Reads& reads) {
        const uint32_t slot = n_proc % nstg;
        const int32_t a = c_lo[slot];
        const int32_t e = c_hi[slot];
        const int32_t s = static_cast<int32_t>(proc_bank);
        const uint32_t base = stage_l1 + slot * chunk_bytes;
        const uint32_t noc = reads.noc.get_noc_id();
        reads.begin(copy_trid);
        ncrisc_noc_read_set_state<DM_DEDICATED_NOC, true>(noc, read_cmd_buf, get_noc_addr(base, noc), pixel_bytes);
        const uint32_t ring_end = ring_l1 + ring_bytes;
        uint32_t src = base + s * slot_bytes;
        int32_t x = (a - origin) % ring_px + s;
        if (x >= ring_px) {
            x -= ring_px;
        }
        uint32_t dst = ring_l1 + static_cast<uint32_t>(x) * pixel_bytes;
        for (int32_t q = a + s; q < e; q += nb) {
            ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, true>(noc, read_cmd_buf, src, dst);
            reads.count_packets(1);
            src += pixel_bytes;
            dst += nb * pixel_bytes;
            if (dst >= ring_end) {
                dst -= ring_bytes;
            }
        }
        reads.drop_fixed_state();
        copies_out = true;
        ++proc_bank;
        if (proc_bank == nb || a + static_cast<int32_t>(proc_bank) >= e) {
            proc_bank = 0;
            ++n_proc;
        }
    }

    // One non-blocking step: take the published window, claim and issue a chunk, de-interleave one landed bank run,
    // or publish `done` once every claimed chunk's copies landed. Leaves the read transaction id at 0.
    template <typename Reads, typename Accessor>
    FORCE_INLINE void serve(Reads& reads, const Accessor& input) {
        if (done) {
            return;
        }
        if (!started) {
            const uint32_t h = rm_load(sw.hi);
            if (h == 0) {
                return;
            }
            hi = static_cast<int32_t>(h);
            lo = static_cast<int32_t>(*sw.lo);
            origin = static_cast<int32_t>(*sw.origin);
            n = static_cast<uint32_t>(hi - lo + chunk_px - 1) / chunk_px;
            started = true;
        }
        if (!met && n_issued - n_proc < nstg) {
            uint32_t j;
            if (claim_back(j)) {
                issue(reads, input, j);
                dataflow_kernel_lib::set_read_trid(reads.noc, 0);
                return;
            }
            met = true;
        }
        if (n_proc < n_issued) {
            if (proc_bank > 0 ||
                ncrisc_noc_read_with_transaction_id_flushed(reads.noc.get_noc_id(), stage_trid0 + n_proc % nstg)) {
                process_bank(reads);
                dataflow_kernel_lib::set_read_trid(reads.noc, 0);
            }
            return;
        }
        if (copies_out && !ncrisc_noc_read_with_transaction_id_flushed(reads.noc.get_noc_id(), copy_trid)) {
            return;
        }
        copies_out = false;
        rm_store(sw.w_done, 1);
        done = true;
    }
};

}  // namespace conv2d_3x3
