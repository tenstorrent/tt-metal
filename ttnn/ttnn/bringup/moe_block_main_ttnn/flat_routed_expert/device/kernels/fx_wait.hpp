// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// MOE_REPLICAS=exact (P.55c, inside routed_experts_combine with the exact copy, REC_FX): a weight reader of the flat
// expert waits for the replica's copy before it reads the spare slot (local expert SE_FX_SPARE). The chip's waiter port
// (combine_port_writer.cpp) writes this call's token into the landed word once every block of the copy has landed in
// this chip's DRAM; the reader polls that word (one 4 B NoC read every ~1 us, only once it reaches the replica, which
// the schedule puts last). Bounded: past `limit` cycles from the kernel's start it reads anyway (never a hang).
//
// Runtime args at SE_FX_RT: on (0: no copy this call - no wait), landed word's core x | y << 16, its address, token,
// limit (cycles), (unused).
#pragma once
#ifdef SE_FX_RT
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

struct SeFxWait {
    uint64_t src = 0;
    uint32_t token = 0, limit = 0, t0 = 0, last = 0, scr = 0;
    bool ok = true, late = false;

    // scratch: 16 B of this RISC's L1, 16 B aligned (the landed word sits at a 16 B boundary)
    void init(uint32_t scratch) {
        ok = get_arg_val<uint32_t>(SE_FX_RT) == 0;
        if (ok) {
            return;
        }
        const uint32_t xy = get_arg_val<uint32_t>(SE_FX_RT + 1);
        src = get_noc_addr(xy & 0xFFFF, xy >> 16, get_arg_val<uint32_t>(SE_FX_RT + 2));
        token = get_arg_val<uint32_t>(SE_FX_RT + 3);
        limit = get_arg_val<uint32_t>(SE_FX_RT + 4);
        scr = scratch;
        t0 = get_timestamp_32b();
        last = t0 - 0x10000000u;
    }

    // true once the copy has landed (or the limit passed); polls at most every ~1 us
    bool ready() {
        if (ok) {
            return true;
        }
        const uint32_t now = get_timestamp_32b();
        if (now - t0 > limit) {
            ok = late = true;
            return true;
        }
        if (now - last < 1350) {
            return false;
        }
        last = now;
        noc_async_read_set_trid(0);
        noc_async_read(src, scr, 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        ok = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr) == token;
        return ok;
    }
};
#endif
