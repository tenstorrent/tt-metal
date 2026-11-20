// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Two-NoC interleaved ROW_MAJOR slab fill (CONV2D_3X3_FILL_NOC_SPLIT; fill_noc_split experiment).
//
// Interleaved DRAM ROW_MAJOR input: every input pixel is one DRAM page read, and the fill is bound by how many reads
// the DRAM endpoints of one NoC accept (bench on 110 cores: ~540 M reads/s over NoC0 alone for 64, 128 and 256 B pages,
// with the top grid rows waiting longest), not by the issuing RISC (the same issue loop from local L1 runs 10x faster).
// Each DRAM bank serves NoC0 and NoC1 through different endpoints, so the writer (BRISC) reads part of every band on
// NoC1 while the reader (NCRISC) reads the rest on NoC0, into the same slab.
//
// The split is claimed per band, so the bands land in order (compute starts kernel row 0 while rows 1-2 land): band
// kh's in-image slab range is cut into CHUNK_PX-pixel chunks; the reader claims them from the front, the writer from
// the back. Each publishes its claim position in its own L1 word (tagged with the block, so a stale word reads as "not
// started"):
//   rpub  (reader): tag | (r + 1), published before the reader checks the writer and issues chunk r (block chunk
//   index); wpub  (writer): tag | (band << 12) | claims in that band, published before the writer issues the claim;
//   wfin  (writer): the writer's final claim count of each band it left (10 bits per band), written before wpub moves
//                   to the next band.
// The writer stops a band GAP chunks short of rpub; the reader runs up to the writer's claims of its band. So no chunk
// is unread, and a chunk is read twice only if the reader advanced GAP chunks between the writer's check and its
// publish (harmless: the same bytes to the same slab pixels, both landed before the band counts as filled).
// A writer that arrives late (it stores the previous block's output first) finds the reader's chunks claimed and moves
// on; the reader then reads more. The reader counts band kh filled once its own reads (NoC0 trid kh + 1) landed, the
// writer published that it left band kh, and the writer's reads of it (NoC1 trid WRITER_TRID0 + kh) landed; it raises
// slab_free at the start of the next block (compute popped every entry pointing into the slab), and the writer claims
// block m only after slab_free reached m.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "conv2d_3x3_slab_dataflow.hpp"

namespace conv2d_3x3::fill_split {

constexpr uint32_t CHUNK_PX = 64;
constexpr uint32_t GAP = 2;
constexpr uint32_t WRITER_TRID0 = 12;  // NoC1 transaction ids of the writer's band reads: 12, 13, 14

// One block's three bands as chunk ranges of in-image slab pixels.
struct Bands {
    int32_t x_lo[3];
    int32_t x_hi[3];
    int32_t v_minus_x[3];
    uint32_t start[4];  // first chunk of band k; start[3] = chunks in the block

    template <int32_t input_width, int32_t num_in_px, int32_t dilation, int32_t band_stride_px>
    FORCE_INLINE void plan(const SlabFootprint& fp, int32_t (&slab_x_base)[3]) {
        start[0] = 0;
        slab_bands<input_width, num_in_px, dilation, band_stride_px>(
            fp, slab_x_base, [&](int32_t kh, int32_t lo, int32_t hi, int32_t vmx) {
                x_lo[kh] = lo;
                x_hi[kh] = hi > lo ? hi : lo;
                v_minus_x[kh] = vmx;
                start[kh + 1] = start[kh] + (static_cast<uint32_t>(x_hi[kh] - lo) + CHUNK_PX - 1) / CHUNK_PX;
            });
    }

    FORCE_INLINE uint32_t band_of(uint32_t c) const { return c < start[1] ? 0 : (c < start[2] ? 1 : 2); }

    // Slab pixels [lo, hi) of chunk c of band k.
    FORCE_INLINE void range(uint32_t c, uint32_t k, int32_t& lo, int32_t& hi) const {
        lo = x_lo[k] + static_cast<int32_t>((c - start[k]) * CHUNK_PX);
        hi = lo + static_cast<int32_t>(CHUNK_PX) < x_hi[k] ? lo + static_cast<int32_t>(CHUNK_PX) : x_hi[k];
    }
};

FORCE_INLINE volatile tt_l1_ptr uint32_t* word(uint32_t sem_id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
}

FORCE_INLINE uint32_t tag_of(uint32_t m_block_idx) { return (m_block_idx + 1) << 16; }

// Reader: claim and issue (issue_chunk(k, lo, hi)) chunks from the front until every chunk is claimed.
template <typename IssueFn>
FORCE_INLINE void reader_claims(
    const Bands& b, uint32_t m_block_idx, uint32_t rpub_id, uint32_t wpub_id, uint32_t wfin_id, IssueFn&& issue_chunk) {
    volatile tt_l1_ptr uint32_t* rpub = word(rpub_id);
    volatile tt_l1_ptr uint32_t* wpub = word(wpub_id);
    volatile tt_l1_ptr uint32_t* wfin = word(wfin_id);
    const uint32_t tag = tag_of(m_block_idx);
    uint32_t r = 0;
    while (r < b.start[3]) {
        *rpub = tag | (r + 1);
        const uint32_t k = b.band_of(r);
        invalidate_l1_cache();
        const uint32_t wv = *wpub;
        uint32_t limit = b.start[k + 1];
#ifdef FNS_STATIC
        // Static variant: the reader takes the first half of every band's chunks, the writer the rest.
        limit = b.start[k] + (b.start[k + 1] - b.start[k] + 1) / 2;
        if (false) {
#else
        if ((wv & 0xFFFF0000u) == tag) {
#endif
            const uint32_t wk = (wv >> 12) & 0xF;
            if (wk == k) {
                limit -= wv & 0xFFF;
            } else if (wk > k) {
                // wfin was written before wpub left band k (the writer fences between them): read it after wpub.
                asm volatile("fence r, r" ::: "memory");
                limit -= (*wfin >> (10 * k)) & 0x3FF;
            }
        }
        if (r >= limit) {
            r = b.start[k + 1];  // the rest of band k is the writer's
            continue;
        }
        int32_t lo, hi;
        b.range(r, k, lo, hi);
        issue_chunk(k, lo, hi);
        ++r;
    }
}

// Reader: band kh of this block is filled on the writer's side (it left the band and its reads landed).
FORCE_INLINE void wait_writer_band(uint32_t kh, uint32_t m_block_idx, uint32_t wpub_id) {
    volatile tt_l1_ptr uint32_t* wpub = word(wpub_id);
    const uint32_t tag = tag_of(m_block_idx);
    while (true) {
        invalidate_l1_cache();
        const uint32_t wv = *wpub;
        if ((wv & 0xFFFF0000u) == tag && ((wv >> 12) & 0xF) > kh) {
            break;
        }
    }
    while (!ncrisc_noc_read_with_transaction_id_flushed(1 - noc_index, WRITER_TRID0 + kh)) {
    }
}

// Writer: per band, claim and issue chunks from the back (issue_chunk(k, lo, hi), under the band's NoC1 trid) until
// GAP chunks short of the reader, then publish the band's final count and move on.
template <typename IssueFn>
FORCE_INLINE void writer_claims(
    const Bands& b, uint32_t m_block_idx, uint32_t rpub_id, uint32_t wpub_id, uint32_t wfin_id, IssueFn&& issue_chunk) {
    volatile tt_l1_ptr uint32_t* rpub = word(rpub_id);
    volatile tt_l1_ptr uint32_t* wpub = word(wpub_id);
    volatile tt_l1_ptr uint32_t* wfin = word(wfin_id);
    const uint32_t tag = tag_of(m_block_idx);
    uint32_t fin = 0;
    for (uint32_t k = 0; k < 3; ++k) {
        *wpub = tag | (k << 12);
        noc_async_read_set_trid(WRITER_TRID0 + k);
        uint32_t w = 0;
        while (b.start[k] + w < b.start[k + 1]) {
            const uint32_t c = b.start[k + 1] - 1 - w;
            invalidate_l1_cache();
            const uint32_t rv = *rpub;
            const uint32_t r = (rv & 0xFFFF0000u) == tag ? (rv & 0xFFFFu) : 0;
#ifdef FNS_STATIC
            (void)r;
            if (c < b.start[k] + (b.start[k + 1] - b.start[k] + 1) / 2) {
                break;
            }
#else
            if (c < r + GAP) {
                break;
            }
#endif
            *wpub = tag | (k << 12) | (w + 1);
            int32_t lo, hi;
            b.range(c, k, lo, hi);
            issue_chunk(k, lo, hi);
            ++w;
        }
        fin |= w << (10 * k);
        *wfin = fin;
        // The reader reads band k's final count only once wpub left band k: order the two stores.
        asm volatile("fence w, w" ::: "memory");
    }
    *wpub = tag | (3u << 12);
    noc_async_read_set_trid(0);
}

}  // namespace conv2d_3x3::fill_split
