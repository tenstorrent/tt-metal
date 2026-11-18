// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Gate/up weights read by the gate/up core itself (on NOC0 from its NCRISC, gate_up_x_forwarder.cpp), through a grant /
// landing-slot protocol with the receiver. Chunk c of the call (load
// l = c / PER_E, chunk load_chunk = c % PER_E) is read from this core's slot bank once the receiver has granted it (the
// receiver's grant counter >= c + 1) into landing slot (region of l) x PER_E + load_chunk, DEPTH reads in flight
// (transaction ids TRID0 ..), and the receiver's DATA count is set as chunks land, in order: the receiver loop
// (gate_up_core_reader.cpp) and the compute publish blocks by that count.
//
// Runtime args (kGateUpReadArgs from the index init takes): 0 gate/up weights address (bank-local), 1 bank id, 2 this
// core's first block in the bank (slot region + block j), 3 chunk bytes (RG blocks), 4 landing ring address
#pragma once
#include <stdint.h>
#include "../../../common/kernels/moe_block_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "flat_options.hpp"

constexpr uint32_t kGateUpReadArgs = 5;
constexpr uint32_t kGateUpReadFirstTrid = 8;  // (x forwarding writes use ids 1..; the h writes id 0)

// The loads list a BRISC hands its NCRISC: [loads, load l's expert | region << 8 ..] in the NCRISC's half of CB 7,
// then the local semaphore flat_options::gate_up_mailbox_sem set (zero at launch)
struct GateUpWeightReads {
    uint32_t base = 0, bank = 0, chunk = 0, block_bytes = 0, per_e = 0, land = 0, depth = 1;
    uint32_t total = 0, issued = 0, done = 0, issue_trid = 0, retire_trid = 0, l = 0, load_chunk = 0;
    // a block read as PIECES reads of block_bytes / PIECES (DEPTH counts pieces in flight); issued / done count pieces,
    // piece_index the next piece of the block being issued
    static constexpr uint32_t pieces = flat_options::gate_up_read_pieces;
    uint32_t pbytes = 0, piece_index = 0;
    volatile tt_l1_ptr uint32_t* grant = nullptr;
    volatile tt_l1_ptr uint32_t* data = nullptr;
    const volatile tt_l1_ptr uint32_t* loads = nullptr;  // expert | region << 8 a load

    void init(
        uint32_t first_arg,
        uint32_t blk_bytes,
        uint32_t per_e_,
        uint32_t depth_,
        uint32_t n_load,
        const volatile tt_l1_ptr uint32_t* loads_,
        volatile tt_l1_ptr uint32_t* grant_,
        volatile tt_l1_ptr uint32_t* data_) {
        base = moe_block_args::arg(first_arg) + moe_block_args::arg(first_arg + 2);
        bank = moe_block_args::arg(first_arg + 1);
        chunk = moe_block_args::arg(first_arg + 3);
        land = moe_block_args::arg(first_arg + 4);
        block_bytes = blk_bytes;
        per_e = per_e_;
        depth = depth_;
        pbytes = block_bytes / pieces;
        total = n_load * per_e * pieces;
        loads = loads_;
        grant = grant_;
        data = data_;
    }

    // Issues the granted reads, publishes the landed ones; true once every chunk has landed (call with the L1 cache
    // invalidated: the grant counter is the receiver's)
    bool step() {
        while (issued < total && issued - done < depth && *grant >= issued / pieces + 1) {
            const uint32_t v = loads[l];
            const uint32_t src = base + ((v & 0xFF) * per_e + load_chunk) * chunk + piece_index * pbytes;
            const uint32_t dst = land + (((v >> 8) & 0xFF) * per_e + load_chunk) * block_bytes + piece_index * pbytes;
            noc_async_read_set_trid(kGateUpReadFirstTrid + issue_trid);
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, src), dst, pbytes);
            ++issued;
            issue_trid = issue_trid + 1 == depth ? 0 : issue_trid + 1;
            if (++piece_index == pieces) {
                piece_index = 0;
                if (++load_chunk == per_e) {
                    load_chunk = 0;
                    ++l;
                }
            }
        }
        // (the read command buffer drained first: an id's outstanding count covers only requests the NIU has taken)
        while (done < issued && !noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
        }
        bool moved = false;
        while (done < issued &&
               ncrisc_noc_read_with_transaction_id_flushed(noc_index, kGateUpReadFirstTrid + retire_trid)) {
            ++done;
            retire_trid = retire_trid + 1 == depth ? 0 : retire_trid + 1;
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
