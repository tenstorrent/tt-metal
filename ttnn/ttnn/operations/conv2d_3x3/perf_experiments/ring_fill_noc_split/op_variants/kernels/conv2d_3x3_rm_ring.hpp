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
// Split fill (RM_SPLIT, `split`): the writer (BRISC, NoC1) fills part of the stream through its own staging chunks
// (RmRingWriterPart). The reader cuts the stream into epochs: each time `allowed` lets the fill go further and the
// previous epoch is fully claimed, the pixels [next stream pixel, min(allowed, its segment's end)) become epoch e,
// published in L1 words (a sequence lock: odd while the range is rewritten). Both RISCs number an epoch's chunks from
// its first pixel; the reader claims them from the front, the writer from the back, until they meet, so each core's
// split follows whichever RISC and NoC deliver faster there. A claim is a Dekker pair: each side publishes its claim
// count, then reads the other's. The writer marks each claim undecided until it has read the reader's word; it gives
// the chunk up if the reader claimed it, and the reader reads only decided writer claims, so every chunk has exactly
// one owner. The writer publishes done = e once its chunks of epoch e landed in the ring (it starts a later epoch only
// after that); the reader's `landed` passes the start of the latest epoch's writer part only once done >= that epoch.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "conv2d_3x3_slab_dataflow.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"

namespace conv2d_3x3 {

constexpr int32_t RM_INF = 0x7FFFFFFF;

// Split fill handshake words (L1 semaphores of this core). Claim words: (epoch << 16) | count; the writer's also
// carries RM_DECIDED once it checked its latest claim against the reader's word.
struct RmSplitWords {
    volatile tt_l1_ptr uint32_t* ep_seq;  // 2 e once epoch e's range is published, 2 e - 1 while it is written
    volatile tt_l1_ptr uint32_t* ep_lo;
    volatile tt_l1_ptr uint32_t* ep_hi;
    volatile tt_l1_ptr uint32_t* r_claim;
    volatile tt_l1_ptr uint32_t* w_claim;
    volatile tt_l1_ptr uint32_t* w_done;  // the last epoch whose writer chunks landed
    volatile tt_l1_ptr uint32_t* origin;  // the ring origin (the reader publishes it before epoch 1)

    template <uint32_t seq, uint32_t lo, uint32_t hi, uint32_t rc, uint32_t wc, uint32_t done, uint32_t org>
    static FORCE_INLINE RmSplitWords from_semaphores() {
        auto w = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
        return RmSplitWords{w(seq), w(lo), w(hi), w(rc), w(wc), w(done), w(org)};
    }
};
constexpr uint32_t RM_DECIDED = 0x8000;
constexpr uint32_t RM_COUNT_MASK = 0x7FFF;

// The other RISC's word (past this RISC's L1 cache).
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
    bool split = false>
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
    // Split fill: the handshake words; the current epoch (0: none yet), its pixels [ep_lo, ep_hi), chunks and the
    // reader's front claims; whether it is still claimable; the stream position after it (iss_seg / iss_a then point
    // at the next epoch's first pixel); the latest epoch with a writer part not yet seen done, and that part's first
    // pixel (RM_INF: none pending).
    RmSplitWords sw{};
    uint32_t ep = 0;
    int32_t ep_lo = 0;
    int32_t ep_hi = 0;
    uint32_t ep_n = 0;
    uint32_t ep_rc = 0;
    bool ep_open = false;
    int32_t ep_next = 0;
    uint32_t wp_ep = 0;
    int32_t wp_lo = RM_INF;

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
            rm_store(sw.origin, static_cast<uint32_t>(origin));
            open_epoch();
        }
    }

    // Split fill: publish the next epoch, [iss_a, min(allowed, its segment's end)), if the stream goes on below
    // `allowed`. The reader's claim word is reset first, so the writer never pairs the new range with an old count.
    FORCE_INLINE bool open_epoch() {
        if (ep_open || iss_seg >= num_segs) {
            return false;
        }
        const int32_t lo = iss_a;
        const int32_t s_hi = seg_hi[iss_seg];
        const int32_t hi = s_hi < allowed ? s_hi : allowed;
        if (lo >= hi) {
            return false;
        }
        ++ep;
        ep_lo = lo;
        ep_hi = hi;
        ep_n = static_cast<uint32_t>(hi - lo + chunk_px - 1) / chunk_px;
        ASSERT(ep < 0x10000 && ep_n <= RM_COUNT_MASK);
        ep_rc = 0;
        ep_open = true;
        rm_store(sw.r_claim, ep << 16);
        rm_store(sw.ep_seq, 2 * ep - 1);
        *sw.ep_lo = static_cast<uint32_t>(lo);
        rm_store(sw.ep_hi, static_cast<uint32_t>(hi));
        rm_store(sw.ep_seq, 2 * ep);
        iss_a = hi;
        if (hi == s_hi) {
            ++iss_seg;
            iss_a = iss_seg < num_segs ? seg_lo[iss_seg] : RM_INF;
        }
        ep_next = iss_a;
        return true;
    }

    // The writer's claims in the current epoch, once its latest one is decided.
    FORCE_INLINE uint32_t writer_claims() const {
        uint32_t w;
        do {
            w = rm_load(sw.w_claim);
        } while ((w >> 16) == ep && (w & RM_DECIDED) == 0);
        return (w >> 16) == ep ? (w & RM_COUNT_MASK) : 0;
    }

    // The front and back claims met: the writer owns the epoch's last wc chunks. Every pixel from the reader's last
    // chunk to ep_next is then the writer's (or in a segment gap nothing reads).
    FORCE_INLINE void close_epoch(uint32_t wc) {
        ep_open = false;
        if (wc > 0) {
            // The writer finished any earlier epoch before it claimed in this one.
            if (wp_lo < RM_INF) {
                while (rm_load(sw.w_done) < wp_ep) {
                }
            }
            wp_ep = ep;
            wp_lo = ep_lo + static_cast<int32_t>(ep_n - wc) * chunk_px;
        }
        if (n_issued > n_proc) {
            c_after[(n_issued - 1) % nstg] = ep_next;
        } else {
            copied = ep_next;
        }
    }

    // The next front chunk of the current epoch (opening one if the last is closed), or false: nothing claimable now.
    FORCE_INLINE bool claim_front() {
        while (true) {
            if (!ep_open && !open_epoch()) {
                return false;
            }
            uint32_t wc = writer_claims();
            if (ep_rc + wc < ep_n) {
                rm_store(sw.r_claim, (ep << 16) | (ep_rc + 1));
                wc = writer_claims();
                if (ep_rc + wc < ep_n) {
                    ++ep_rc;
                    return true;
                }
            }
            close_epoch(wc);
        }
    }

    // The first pixel the reader's next claim would cover.
    FORCE_INLINE int32_t front_px() const { return ep_open ? ep_lo + static_cast<int32_t>(ep_rc) * chunk_px : iss_a; }

    // Every needed pixel below it has landed, the writer's part included.
    FORCE_INLINE int32_t landed_all() {
        if constexpr (split) {
            if (wp_lo < RM_INF) {
                if (rm_load(sw.w_done) < wp_ep) {
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
        if constexpr (split) {
            // Publish the newly allowed pixels at once, so the writer can start on them.
            if (ep_open) {
                const uint32_t wc = writer_claims();
                if (ep_rc + wc >= ep_n) {
                    close_epoch(wc);
                }
            }
            if (!ep_open) {
                open_epoch();
            }
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

    // Split fill: the DRAM reads of the front chunk just claimed.
    template <typename Reads, typename Accessor>
    FORCE_INLINE void issue_claimed(Reads& reads, const Accessor& input) {
        const uint32_t slot = n_issued % nstg;
        if (n_issued >= nstg) {
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
        if constexpr (split) {
            if (n_issued - n_proc < nstg && claim_front()) {
                issue_claimed(reads, input);
                return true;
            }
            if (n_proc < n_issued &&
                ncrisc_noc_read_with_transaction_id_flushed(reads.noc.get_noc_id(), stage_trid0 + n_proc % nstg)) {
                process(reads);
                return true;
            }
            return false;
        }
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

    // Blocking: every needed pixel below x is in the ring. x is within the plan's capacity of the live pixels
    // (x <= allowed), so the chunks it needs can be issued.
    template <typename Reads, typename Accessor>
    FORCE_INLINE void need(Reads& reads, const Accessor& input, int32_t x) {
        if constexpr (split) {
            if (x <= landed_all()) {
                return;
            }
            ASSERT(x <= allowed);
            while (copied < x) {
                if (n_issued - n_proc < nstg && front_px() < x && claim_front()) {
                    issue_claimed(reads, input);
                } else if (n_proc < n_issued) {
                    process(reads);
                } else if (!ep_open) {
                    // The last epoch closed with nothing staged (copied moved past its writer part): the next one.
                    open_epoch();
                }
            }
            land_copies(reads);
            if (landed_all() < x) {
                MaybeDeviceZoneScope("reader_rm_split_wait");
                while (rm_load(sw.w_done) < wp_ep) {
                }
                wp_lo = RM_INF;
            }
            return;
        }
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

// The split fill's writer part (BRISC, NoC1): claims the current epoch's chunks from the back, one DRAM read per bank
// into its own nstg staging chunks, then one self-aimed read per pixel into the ring (as RmBankRing). The writer calls
// serve() while it waits for output tiles, so one call does at most one bank's de-interleave run (chunk_px / nb
// pixels): a long chunk copy would hold back the output store and stall compute's bias pass on the output CB.
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
    int32_t origin = -1;  // read once the reader published epoch 1
    RmSplitWords sw{};
    // The epoch being filled (or the last one done), its pixels and chunks, this writer's back claims.
    uint32_t ep = 0;
    bool active = false;
    bool met = false;
    int32_t lo = 0;
    int32_t hi = 0;
    uint32_t n = 0;
    uint32_t wc = 0;
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

    // Claim the next chunk from the back (false: the claims met, or the reader moved past this epoch).
    FORCE_INLINE bool claim_back(uint32_t& j) {
        if (wc >= n) {
            return false;
        }
        j = n - 1 - wc;
        uint32_t r = rm_load(sw.r_claim);
        if ((r >> 16) != ep || (r & 0xFFFFu) > j) {
            return false;
        }
        rm_store(sw.w_claim, (ep << 16) | (wc + 1));
        r = rm_load(sw.r_claim);
        if ((r >> 16) != ep || (r & 0xFFFFu) > j) {
            rm_store(sw.w_claim, (ep << 16) | RM_DECIDED | wc);
            return false;
        }
        ++wc;
        rm_store(sw.w_claim, (ep << 16) | RM_DECIDED | wc);
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

    // One non-blocking step: take a newly published epoch, claim and issue a chunk, de-interleave a landed one, or
    // publish the epoch done once every claimed chunk's copies landed. Leaves the read transaction id at 0.
    template <typename Reads, typename Accessor>
    FORCE_INLINE void serve(Reads& reads, const Accessor& input) {
        if (!active) {
            const uint32_t s1 = rm_load(sw.ep_seq);
            if ((s1 & 1u) != 0 || s1 / 2 <= ep) {
                return;
            }
            const int32_t l = static_cast<int32_t>(*sw.ep_lo);
            const int32_t h = static_cast<int32_t>(*sw.ep_hi);
            if (rm_load(sw.ep_seq) != s1) {
                return;
            }
            if (origin < 0) {
                origin = static_cast<int32_t>(rm_load(sw.origin));
            }
            ep = s1 / 2;
#if defined(RM_SPLIT_W_EPOCHS) && RM_SPLIT_W_EPOCHS > 0
            if (ep > RM_SPLIT_W_EPOCHS) {
                return;
            }
#endif
            lo = l;
            hi = h;
            n = static_cast<uint32_t>(h - l + chunk_px - 1) / chunk_px;
            wc = 0;
            met = false;
            active = true;
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
        if (met) {
            if (copies_out) {
                if (!ncrisc_noc_read_with_transaction_id_flushed(reads.noc.get_noc_id(), copy_trid)) {
                    return;
                }
                copies_out = false;
            }
            rm_store(sw.w_done, ep);
            active = false;
        }
    }
};

}  // namespace conv2d_3x3
