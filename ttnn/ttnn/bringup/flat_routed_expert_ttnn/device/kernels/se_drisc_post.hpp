// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Posting a flat call's DRISC jobs (se_drisc.hpp) from a Tensix core: the launcher (se_drisc_go.cpp), or, with dual
// slots, the BRISC of DRISC slot 0's own Tensix reader (se_reader.cpp SE_DRISC_LAUNCH; it idles on DRISC calls). From
// the call's schedule: one job descriptor per slot written into its DRISC's L1, then the DRISC's job counter bumped.
//
// Runtime args from A0: 0 DRISC working base, 1 gate/up weights address, 2 landing (arena), 3 NS slots, 4 R (the
// receivers' stride: the most of any slot), 5 MPFE word (SD_MPFE), 6 (unused), 7 BLK_BYTES, 8 (unused), 9 G, 10 PER_E,
// 11 GRANT_SEM id, 12 DATA_SEM id, then per slot (SE_DRISC_SLOT_ARGS): DRISC xy, job index on it, region offset in its
// bank, alternate DRISC xy (0: none), its job index, its CHUNK_BYTES, RG and receivers (uneven splits: they differ
// per slot); then the slots' receiver xy, R words each. A slot with an alternate is served by it on calls with counts
// <= SE_DRISC_ALT_MAXCNT (else by the first); the one not serving gets a skip (its skip counter bumped, no descriptor).
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"
#include "se_drisc.hpp"

constexpr uint32_t SE_DRISC_SLOT_ARGS = 8;

inline uint32_t se_drisc_post_nargs(uint32_t a0) {
    const uint32_t ns = get_arg_val<uint32_t>(a0 + 3), R = get_arg_val<uint32_t>(a0 + 4);
    return 13 + SE_DRISC_SLOT_ARGS * ns + ns * R;
}

// SCRATCH: 16 B aligned local L1 of SCRATCH_BYTES for the descriptors (the schedule's rows may be reused: dyn is built)
#ifdef SE_DRISC_X
// x args (from XA0, after the post and the se_dyn.hpp args): 0 the launcher's staging (arena; + 256 + slot KB), 1 x
// address, 2 x page bytes, 3 token index address, 4 MT, 5 NSB, 6 VSTRIDE, 7 rectangles, 8 row-major chunks, 9 batch,
// 10 segment bytes, 11 the relays' ring, 12 (unused), 13 banks, 14 slots a bank, 15 DRISC working region bytes, 16..
// the relays' xy (rectangle k, offset o at k * VSTRIDE + o)
inline uint32_t se_drisc_x_stage(const SeDyn& dyn, uint32_t xa0, uint32_t& words, uint32_t& region) {
    auto xa = [&](uint32_t i) { return get_arg_val<uint32_t>(xa0 + i); };
    volatile tt_l1_ptr uint32_t* xi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xa(0) + 256);
    xi[XI_XADDR] = xa(1);
    xi[XI_PAGE] = xa(2);
    xi[XI_MT] = xa(4);
    xi[XI_NSB] = xa(5);
    xi[XI_VSTRIDE] = xa(6);
    xi[XI_NRECT] = xa(7);
    xi[XI_RMC] = xa(8);
    xi[XI_BATCH] = xa(9);
    xi[XI_SEG] = xa(10);
    xi[XI_RING] = xa(11);
    xi[XI_RSV] = get_semaphore(SE_XSEM_RSV);  // (the same address on every core of the program)
    for (uint32_t p = 0; p < XI_MAX_RMC; ++p) {
        xi[XI_ARR + p] = get_semaphore(SE_XSEM_ARR[p]);
    }
    xi[XI_BANKS] = xa(13);
    xi[XI_SPB] = xa(14);
    const InterleavedAddrGen<true> tg = {.bank_base_address = xa(3), .page_size = 4};
    const uint64_t t0 = get_noc_addr(0, tg, 0, 0), t1 = get_noc_addr(0, tg, 0, 1);
    xi[XI_TIDX0] = static_cast<uint32_t>(t0);
    xi[XI_TIDX0 + 1] = static_cast<uint32_t>(t0 >> 32);
    xi[XI_TIDX1] = static_cast<uint32_t>(t1);
    xi[XI_TIDX1 + 1] = static_cast<uint32_t>(t1 >> 32);
    const uint32_t nrl = xa(6) * xa(7);
    for (uint32_t i = 0; i < XI_MAX_RL; ++i) {
        xi[XI_RXY + i] = i < nrl ? xa(16 + i) : 0;
    }
    xi[XI_NACT] = dyn.n_act;
    for (uint32_t a = 0; a < dyn.n_act; ++a) {
        xi[XI_ENT + 2 * a] = dyn.off[a];
        xi[XI_ENT + 2 * a + 1] = dyn.cnt[a];
    }
    words = XI_ENT + 2 * dyn.n_act;
    region = xa(15);
    return xa(0);
}
#endif

inline void se_drisc_post(
    const SeDyn& dyn, uint32_t a0, uint32_t scratch, uint32_t scratch_bytes, [[maybe_unused]] uint32_t xa0 = 0) {
    auto arg = [&](uint32_t i) { return get_arg_val<uint32_t>(a0 + i); };
#ifdef SE_DRISC_X
    uint32_t x_words = 0, x_region = 0;
    const uint32_t x_stage = se_drisc_x_stage(dyn, xa0, x_words, x_region);
#endif
    const uint32_t base = arg(0), gu = arg(1), ns = arg(3), R = arg(4);
    const uint32_t rx0 = a0 + 13 + SE_DRISC_SLOT_ARGS * ns;
    // every slot's descriptor in its own scratch slot (they differ in the region and the receivers), all written,
    // one barrier, then the job counters (and the skips): one round trip per call instead of one per slot. Falls
    // back to one at a time when the descriptors do not fit SCRATCH_BYTES.
#ifdef SE_DRISC_ALT_MAXCNT
    const bool alt_call = dyn.max_cnt <= SE_DRISC_ALT_MAXCNT;
#else
    constexpr bool alt_call = false;
#endif
#ifdef SE_DRISC_DUAL  // the slots also have Tensix readers: they serve calls with larger counts (no DRISC jobs posted:
                      // the DRISCs wait for the next; GDDR MPFE weights stay as the last DRISC call left them)
    if (dyn.max_cnt > SE_DRISC_MAXCNT) {
        return;
    }
#endif
    const uint32_t n_load = dyn.n_load;
    const uint32_t bytes = ((SD_LOADS + n_load) * 4 + 15) / 16 * 16;
    const bool batched = ns * bytes <= scratch_bytes;
#ifndef SE_DRISC_POST_NOACK
    // (batched: the job counts move only after EVERY descriptor's write is acked - one barrier for all of them. The
    // count used to follow its descriptor as soon as the write had left this NIU, on the assumption that the
    // increment on the same static VC lands after the write; a DRISC that sees the count before its descriptor
    // landed reads the previous call's - landing address, receivers, GDDR region. SE_DRISC_POST_NOACK: the old way)
    constexpr uint32_t MAX_POST = 32;
    uint32_t post_xy[MAX_POST], post_job[MAX_POST];
    const bool ack_after = batched && ns <= MAX_POST;
#else
    constexpr bool ack_after = false;
#endif
    for (uint32_t s = 0; s < ns; ++s) {
        const uint32_t sa = 13 + SE_DRISC_SLOT_ARGS * s;
        uint32_t dxy = arg(sa), job = arg(sa + 1);
        const uint32_t axy = arg(sa + 3), ajob = arg(sa + 4), r_s = arg(sa + 7);
        if (axy) {  // the one not serving this call: a skip
            se_drisc_inc(alt_call ? dxy : axy, base + se_drisc_skips(alt_call ? job : ajob), 1);
            if (alt_call) {
                dxy = axy;
                job = ajob;
            }
        }
        const uint32_t buf = scratch + (batched ? s * bytes : 0);
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
        d[SD_CHUNK] = arg(sa + 5);
        d[SD_BLK] = arg(7);
        d[SD_RG] = arg(sa + 6);
        d[SD_R] = r_s;
        d[SD_G] = arg(9);
        d[SD_PER_E] = arg(10);
        d[SD_LAND] = arg(2);
        d[SD_DATA_SEM] = get_semaphore(arg(12));
        d[SD_N_LOAD] = n_load;
        d[SD_GRANTS] = get_semaphore(arg(11));
        d[SD_MPFE] = arg(5);
#ifdef SE_DRISC_X
        d[SD_XON] = 1;
#else
        d[SD_XON] = 0;
#endif
#ifdef SE_DRISC_PAIRED_ALWAYS  // (two tiles per bank every call)
        d[SD_PAIRED] = 1;
#else
        d[SD_PAIRED] = alt_call ? 1 : 0;  // (switch mode: the alternate tiles send next to the free ones)
#endif
        for (uint32_t l = 0; l < n_load; ++l) {
            d[SD_LOADS + l] = dyn.load_eid[l] | (uint32_t(dyn.region[l]) << 8);
        }
        d[SD_GDDR] = gu + arg(sa + 2);
        for (uint32_t j = 0; j < r_s; ++j) {
            d[SD_RXY + j] = get_arg_val<uint32_t>(rx0 + s * R + j);
        }
        se_drisc_write(buf, dxy, base + se_drisc_desc(job), bytes);
#ifdef SE_DRISC_X
        {
            // this slot's x info (se_drisc.hpp XI_*): the common words staged once (se_drisc_x_stage), the slot word
            // per slot - its own staging copy, so one barrier covers them all
            const uint32_t xbuf = x_stage + 256 + s * XI_INFO_BYTES;
            volatile tt_l1_ptr uint32_t* xi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xbuf);
            const volatile tt_l1_ptr uint32_t* x0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(x_stage + 256);
            if (s) {
                for (uint32_t w = 0; w < x_words; ++w) {
                    xi[w] = x0[w];
                }
            }
            xi[XI_SLOT] = s;
            se_drisc_write(xbuf, dxy, base + x_region - (job + 1) * SE_DRISC_XA, (x_words * 4 + 15) / 16 * 16);
        }
#endif
        if (!batched) {
            noc_async_write_barrier();  // (landed before the job count moves; and the buffer is reused)
            se_drisc_inc(dxy, base + se_drisc_jobs(job), 1);
        } else if (ack_after) {
#ifndef SE_DRISC_POST_NOACK
            post_xy[s] = dxy;
            post_job[s] = job;
#endif
        } else {
            // each slot's job count right behind its descriptor: once the write has left this NIU, the increment on
            // the same static VC and route lands after it, so the slot's DRISC starts without waiting for the other
            // slots' descriptors (one barrier for all of them cost ~4.5 us of every call's start)
            while (!ncrisc_noc_nonposted_writes_sent(noc_index)) {
            }
            se_drisc_inc(dxy, base + se_drisc_jobs(job), 1);
        }
    }
#ifndef SE_DRISC_POST_NOACK
    if (ack_after) {  // every descriptor landed (acked), then the job counts
        noc_async_write_barrier();
        for (uint32_t s = 0; s < ns; ++s) {
            se_drisc_inc(post_xy[s], base + se_drisc_jobs(post_job[s]), 1);
        }
    }
#endif
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
