// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end flat expert: x relay primary (NCRISC). Its source is the helpers' tilized super-blocks (MT row tiles x
// SE_SBT K tiles, row tile major) in its landing rings (x read once by the helper relays: super-block b from helper
// b % kXnh, se13_xhelp.cpp): each of a super-block's x blocks ([MT x KBLK] tiles, row tile major in the gate/up
// cores' x ring) goes out as MT unicast pieces into the rectangle's snake head, once the whole super-block's ring
// slots are free there (the head's freed word); the gate/up cores forward it along the snake (se5_recv.cpp /
// se5_xfwd.cpp). The head's XARR moves by a batch's pieces once its writes are acknowledged.
// CT: 0 SB_CB, 1 MT, 2 TILE_BYTES, 3 X_SLOTS, 4 XARR_SEM, 5 KBLK, 6 NUM_E, 7 NSB
// RT: 0 x ring address, 1 snake head xy, 2 the head's freed word address (here), 3 helper 0 xy, 4 landing rings
//     address, 5 landing slots, 6 helper 1 xy, 7.. the se_dyn.hpp args
#include <stdint.h>
constexpr uint32_t kXnh = 2;  // the helper relays feeding a primary
#ifndef SE_SBT
#define SE_SBT 32  // K tiles per super-block (the row-major chunk width / 32 columns)
#endif
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"

void kernel_main() {
    constexpr uint32_t mt = get_compile_time_arg_val(1);
    constexpr uint32_t tb = get_compile_time_arg_val(2);
    constexpr uint32_t x_slots = get_compile_time_arg_val(3);
    constexpr uint32_t xarr_sem = get_compile_time_arg_val(4);
    constexpr uint32_t kblk = get_compile_time_arg_val(5);
    constexpr uint32_t num_e = get_compile_time_arg_val(6), nsb = get_compile_time_arg_val(7);
    constexpr uint32_t per_sb = SE_SBT / kblk;
    constexpr uint32_t sb_tiles = mt * SE_SBT;
    static_assert(SE_SBT % kblk == 0);
    constexpr uint32_t blk_bytes = mt * kblk * tb, piece = kblk * tb;
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t hxy = get_arg_val<uint32_t>(1);
    const uint64_t head_xarr = get_noc_addr(hxy >> 16, hxy & 0xFFFF, get_semaphore(xarr_sem));
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2));
    // the freed word restarts every launch: zeroed here (the head only bumps it after consuming x this relay sent in
    // this launch), so no host write is needed between launches
    *freed = 0;
    const uint32_t land = get_arg_val<uint32_t>(4), land_slots = get_arg_val<uint32_t>(5);
    const uint32_t helper_xy[kXnh] = {get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(6)};
    // Dynamic counts: the super-blocks follow from the active experts' sub-blocks (RT 7.. are the se_dyn.hpp args,
    // CB 7's upper half this RISC's scratch)
    constexpr uint32_t dyn0 = 7;
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, dyn0, get_write_ptr(tt::CBIndex::c_7) + 2 * SE_DYN_HALF, mt * 32);
    const uint32_t num_sb = dyn.num_v * nsb;
    uint32_t sent = 0;
    constexpr uint32_t XMC_CHAIN_DEPTH = 4;  // batches of head writes in flight (transaction ids 1..DEPTH)
    // Batch b's head writes carry transaction id 1 + b % DEPTH; once they are acknowledged the head's XARR moves by
    // that batch's pieces, in batch order (the head counts a contiguous prefix).
    uint32_t cb_iss = 0, cb_done = 0;
    auto chain_retire = [&](uint32_t keep) {  // retire acknowledged batches; block while more than `keep` are out
        // (the write command buffer drained first: an id's outstanding count only covers requests the NIU has taken -
        // right after issuing, the last write can still sit in the command buffer and the id reads "done" early)
        while (cb_done < cb_iss && !noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        }
        while (cb_done < cb_iss && (cb_iss - cb_done > keep || ncrisc_noc_nonposted_write_with_transaction_id_flushed(
                                                                   noc_index, 1 + cb_done % XMC_CHAIN_DEPTH))) {
            while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + cb_done % XMC_CHAIN_DEPTH)) {
            }
            noc_semaphore_inc(head_xarr, per_sb * XMC_CHAIN_PIECES);
            ++cb_done;
        }
    };
    for (uint32_t b = 0; b < num_sb; ++b) {
        // Super-block b was read and tilized by helper h = b % kXnh (se13_xhelp.cpp) into landing ring h here (ring h
        // follows ring h - 1); DATA sem help_sem[h] here counts arrivals, the CREDIT sem XMC_CREDIT_SEM on the helper
        // counts freed slots.
        const uint32_t hh = b % kXnh, hb = b / kXnh;
        constexpr uint32_t help_sem[kXnh] = {4, 7};
        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(help_sem[hh])) < hb + 1) {
            invalidate_l1_cache();
            chain_retire(XMC_CHAIN_DEPTH);  // (waiting: count acknowledged batches into the head's XARR)
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(help_sem[hh])), hb + 1);
        const uint32_t src = land + (hh * land_slots + hb % land_slots) * sb_tiles * tb;
        // The whole super-block goes out as one batch once its last block's ring slot is free. Batches in flight are
        // only counted into the head's XARR on retirement: every wait retires the acknowledged ones, else a compute
        // that needs the whole ring (row passes: all Ht / KBLK blocks of a sub-block resident, GLM-5.3 24 of 24)
        // never sees the last batch and never frees one (deadlock).
        invalidate_l1_cache();
        while (sent + per_sb > *freed + x_slots) {
            chain_retire(XMC_CHAIN_DEPTH);
            invalidate_l1_cache();
        }
        // the batch's pieces into the snake head (unicast, under the batch's transaction id); the head forwards them
        // along the snake, se5_recv.cpp
        chain_retire(XMC_CHAIN_DEPTH - 1);  // (room for this batch's id)
        for (uint32_t k = 0; k < per_sb; ++k) {
            const uint32_t dst = ring + ((sent + k) % x_slots) * blk_bytes;
            for (uint32_t m = 0; m < mt; ++m) {
                noc_async_write_one_packet_with_trid(
                    src + (m * SE_SBT + k * kblk) * tb,
                    get_noc_addr(hxy >> 16, hxy & 0xFFFF, dst + m * piece),
                    piece,
                    1 + cb_iss % XMC_CHAIN_DEPTH);
            }
        }
        ++cb_iss;
        chain_retire(XMC_CHAIN_DEPTH);
        sent += per_sb;
        noc_async_writes_flushed();
        noc_semaphore_inc(
            get_noc_addr(helper_xy[hh] >> 16, helper_xy[hh] & 0xFFFF, get_semaphore(XMC_CREDIT_SEM)),
            1);  // slot free
    }
    chain_retire(0);
    noc_async_write_set_trid(0);
    noc_async_write_barrier();
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
