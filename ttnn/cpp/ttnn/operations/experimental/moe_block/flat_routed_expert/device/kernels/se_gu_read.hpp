// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Gate/up weights read by the gate/up core itself (on NOC0 from its NCRISC, se5_xfwd.cpp), through a grant /
// landing-slot protocol with the receiver. Chunk c of the call (load
// l = c / PER_E, chunk ce = c % PER_E) is read from this core's slot bank once the receiver has granted it (the
// receiver's grant counter >= c + 1) into landing slot (region of l) x PER_E + ce, DEPTH reads in flight (transaction
// ids TRID0 ..), and the receiver's DATA count is set as chunks land, in order: the receiver loop (se5_recv.cpp) and
// the compute publish blocks by that count.
//
// Runtime args (SE_GU_RT_ARGS from A0): 0 gate/up weights address (bank-local), 1 bank id, 2 this core's first block
// in the bank (slot region + block j), 3 chunk bytes (RG blocks), 4 landing ring address
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t SE_GU_RT_ARGS = 5;
constexpr uint32_t SE_GU_TRID0 = 8;  // (x forwarding writes use ids 1..; the h writes id 0)

// The loads list a BRISC hands its NCRISC: [n_load, load l's expert | region << 8 ..] in the NCRISC's half of CB 7,
// then the local word SE_GU_MBOX_SEM set (zero at launch)
struct SeGuRead {
    uint32_t base = 0, bank = 0, chunk = 0, blk = 0, per_e = 0, land = 0, depth = 1;
    uint32_t total = 0, iss = 0, done = 0, ri = 0, rd = 0, l = 0, ce = 0;
    // a block read as PIECES reads of blk / PIECES (DEPTH counts pieces in flight); iss / done count pieces, pc the
    // next piece of the block being issued
    static constexpr uint32_t pieces = SE_GU_PIECES;
    uint32_t pbytes = 0, pc = 0;
    volatile tt_l1_ptr uint32_t* grant = nullptr;
    volatile tt_l1_ptr uint32_t* data = nullptr;
    const volatile tt_l1_ptr uint32_t* loads = nullptr;  // expert | region << 8 a load

    void init(
        uint32_t a0,
        uint32_t blk_bytes,
        uint32_t per_e_,
        uint32_t depth_,
        uint32_t n_load,
        const volatile tt_l1_ptr uint32_t* loads_,
        volatile tt_l1_ptr uint32_t* grant_,
        volatile tt_l1_ptr uint32_t* data_) {
        base = get_arg_val<uint32_t>(a0) + get_arg_val<uint32_t>(a0 + 2);
        bank = get_arg_val<uint32_t>(a0 + 1);
        chunk = get_arg_val<uint32_t>(a0 + 3);
        land = get_arg_val<uint32_t>(a0 + 4);
        blk = blk_bytes;
        per_e = per_e_;
        depth = depth_;
        pbytes = blk / pieces;
        total = n_load * per_e * pieces;
        loads = loads_;
        grant = grant_;
        data = data_;
    }

    // Issues the granted reads, publishes the landed ones; true once every chunk has landed (call with the L1 cache
    // invalidated: the grant counter is the receiver's)
    bool step() {
        while (iss < total && iss - done < depth && *grant >= iss / pieces + 1) {
            const uint32_t v = loads[l];
            const uint32_t src = base + ((v & 0xFF) * per_e + ce) * chunk + pc * pbytes;
            const uint32_t dst = land + (((v >> 8) & 0xFF) * per_e + ce) * blk + pc * pbytes;
            noc_async_read_set_trid(SE_GU_TRID0 + ri);
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, src), dst, pbytes);
            ++iss;
            ri = ri + 1 == depth ? 0 : ri + 1;
            if (++pc == pieces) {
                pc = 0;
                if (++ce == per_e) {
                    ce = 0;
                    ++l;
                }
            }
        }
        // (the read command buffer drained first: an id's outstanding count covers only requests the NIU has taken)
        while (done < iss && !noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
        }
        bool moved = false;
        while (done < iss && ncrisc_noc_read_with_transaction_id_flushed(noc_index, SE_GU_TRID0 + rd)) {
            ++done;
            rd = rd + 1 == depth ? 0 : rd + 1;
            moved = true;
        }
        if (moved) {
            *data = done / pieces;
        }
        return done == total;
    }

    // the read command buffer's transaction id back to 0 and every read retired (the next program starts clean)
    void finish() {
        noc_async_read_set_trid(0);
        noc_async_read_barrier();
    }
};
