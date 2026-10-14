// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Posting a flat call's DRISC jobs (se_drisc.hpp) from a Tensix core: the launcher (se_drisc_go.cpp), or, with dual
// slots, the BRISC of DRISC slot 0's own Tensix reader (se_reader.cpp SE_DRISC_LAUNCH; it idles on DRISC calls). From
// the call's schedule: one job descriptor per slot written into its DRISC's L1, then the DRISC's job counter bumped.
//
// Runtime args from A0: 0 DRISC working base, 1 gate/up weights address, 2 landing (arena), 3 NS slots, 4 R, 5 MPFE
// word (SD_MPFE), 6 CHUNK_BYTES, 7 BLK_BYTES, 8 RG, 9 G, 10 PER_E, 11 GRANT_SEM id, 12 DATA_SEM id, then per slot:
// DRISC xy, job index on it, region offset in its bank; then the slots' R receiver xy each.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"
#include "se_drisc.hpp"

inline uint32_t se_drisc_post_nargs(uint32_t a0) {
    const uint32_t ns = get_arg_val<uint32_t>(a0 + 3), R = get_arg_val<uint32_t>(a0 + 4);
    return 13 + 3 * ns + ns * R;
}

// SCRATCH: 16 B aligned local L1 for one descriptor (the schedule's rows may be reused: dyn is built)
inline void se_drisc_post(const SeDyn& dyn, uint32_t a0, uint32_t scratch) {
    auto arg = [&](uint32_t i) { return get_arg_val<uint32_t>(a0 + i); };
    const uint32_t base = arg(0), gu = arg(1), ns = arg(3), R = arg(4);
    const uint32_t rx0 = a0 + 13 + 3 * ns;
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    d[SD_CHUNK] = arg(6);
    d[SD_BLK] = arg(7);
    d[SD_RG] = arg(8);
    d[SD_R] = R;
    d[SD_G] = arg(9);
    d[SD_PER_E] = arg(10);
    d[SD_LAND] = arg(2);
    d[SD_DATA_SEM] = get_semaphore(arg(12));
#ifdef SE_DRISC_DUAL  // the slots also have Tensix readers: they serve calls with larger counts (no DRISC jobs posted:
                      // the DRISCs wait for the next; GDDR MPFE weights stay as the last DRISC call left them)
    if (dyn.max_cnt > SE_DRISC_MAXCNT) {
        return;
    }
#endif
    constexpr bool drisc_call = true;
    const uint32_t n_load = drisc_call ? dyn.n_load : 0;
    d[SD_N_LOAD] = n_load;
    d[SD_GRANTS] = get_semaphore(arg(11));
    d[SD_MPFE] = drisc_call ? arg(5) : 0;
    for (uint32_t l = 0; l < n_load; ++l) {
        d[SD_LOADS + l] = dyn.load_eid[l] | (uint32_t(dyn.region[l]) << 8);
    }
    const uint32_t bytes = ((SD_LOADS + n_load) * 4 + 15) / 16 * 16;
    for (uint32_t s = 0; s < ns; ++s) {  // the slots' descriptors differ in the region and the receivers only
        const uint32_t dxy = arg(13 + 3 * s), job = arg(14 + 3 * s);
        d[SD_GDDR] = gu + arg(15 + 3 * s);
        for (uint32_t j = 0; j < R; ++j) {
            d[SD_RXY + j] = get_arg_val<uint32_t>(rx0 + s * R + j);
        }
        se_drisc_write(scratch, dxy, base + se_drisc_desc(job), bytes);
        noc_async_write_barrier();  // (landed before the job count moves; and the buffer is reused)
        se_drisc_inc(dxy, base + se_drisc_jobs(job), 1);
    }
    noc_async_atomic_barrier();
}
