// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Posting a flat call's DRISC jobs (se_drisc.hpp) from a Tensix core, the launcher (se_drisc_go.cpp). From the call's
// schedule: one job descriptor per slot written into its DRISC's L1, then the DRISC's job counter bumped.
//
// Runtime args from A0: 0 DRISC working base, 1 gate/up weights address, 2 landing (arena), 3 NS slots, 4 R (the
// receivers' stride: the most of any slot), 5 MPFE word (SD_MPFE), 6 DMA reads in flight a job (SD_DMA_CAP), 7
// BLK_BYTES, 8 blocks a unit (SD_UB), 9 G, 10 PER_E, 11 GRANT_SEM id, 12 DATA_SEM id, then per slot
// (SE_DRISC_SLOT_ARGS): DRISC xy, job index on it, region offset in its bank, alternate DRISC xy (0: none), its job
// index, its CHUNK_BYTES, RG and receivers (uneven splits: they differ per slot); then the slots' receiver xy, R words
// each. A slot with an alternate (switch mode) is served by it on every call; its first DRISC's job gets a skip (its
// skip counter bumped, no descriptor).
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
    const uint32_t n_load = dyn.n_load;
    const uint32_t bytes = ((SD_LOADS + n_load) * 4 + 15) / 16 * 16;
    const bool batched = ns * bytes <= scratch_bytes;
    // (batched: the job counts move only after EVERY descriptor's write is acked - one barrier for all of them. The
    // count used to follow its descriptor as soon as the write had left this NIU, on the assumption that the
    // increment on the same static VC lands after the write; a DRISC that sees the count before its descriptor
    // landed reads the previous call's - landing address, receivers, GDDR region)
    constexpr uint32_t MAX_POST = 32;
    uint32_t post_xy[MAX_POST], post_job[MAX_POST];
    const bool ack_after = batched && ns <= MAX_POST;
    for (uint32_t s = 0; s < ns; ++s) {
        const uint32_t sa = 13 + SE_DRISC_SLOT_ARGS * s;
        uint32_t dxy = arg(sa), job = arg(sa + 1);
        const uint32_t axy = arg(sa + 3), ajob = arg(sa + 4), r_s = arg(sa + 7);
        if (r_s == 0) {  // no receivers: the slot's gate/up cores read it themselves (hybrid plans); a skip
            se_drisc_inc(dxy, base + se_drisc_skips(job), 1);
            if (ack_after) {
                post_xy[s] = 0;
            }
            continue;
        }
        if (axy) {  // the alternate serves: the first DRISC's job a skip
            se_drisc_inc(dxy, base + se_drisc_skips(job), 1);
            dxy = axy;
            job = ajob;
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
        d[SD_UB] = arg(8);
        d[SD_DMA_CAP] = arg(6);
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
        } else if (ack_after) {
            post_xy[s] = dxy;
            post_job[s] = job;
        } else {
            // each slot's job count right behind its descriptor: once the write has left this NIU, the increment on
            // the same static VC and route lands after it, so the slot's DRISC starts without waiting for the other
            // slots' descriptors (one barrier for all of them cost ~4.5 us of every call's start)
            while (!ncrisc_noc_nonposted_writes_sent(noc_index)) {
            }
            se_drisc_inc(dxy, base + se_drisc_jobs(job), 1);
        }
    }
    if (ack_after) {  // every descriptor landed (acked), then the job counts
        noc_async_write_barrier();
        for (uint32_t s = 0; s < ns; ++s) {
            if (post_xy[s]) {
                se_drisc_inc(post_xy[s], base + se_drisc_jobs(post_job[s]), 1);
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
