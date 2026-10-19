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
inline void se_drisc_post(const SeDyn& dyn, uint32_t a0, uint32_t scratch, uint32_t scratch_bytes) {
    auto arg = [&](uint32_t i) { return get_arg_val<uint32_t>(a0 + i); };
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
    uint32_t tgt_xy[16], tgt_job[16];
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
        if (!batched) {
            noc_async_write_barrier();  // (landed before the job count moves; and the buffer is reused)
            se_drisc_inc(dxy, base + se_drisc_jobs(job), 1);
        } else {
            tgt_xy[s] = dxy;
            tgt_job[s] = job;
        }
    }
    if (batched) {
        noc_async_write_barrier();  // (every descriptor landed before any job count moves)
        for (uint32_t s = 0; s < ns; ++s) {
            se_drisc_inc(tgt_xy[s], base + se_drisc_jobs(tgt_job[s]), 1);
        }
    }
    noc_async_atomic_barrier();
}
