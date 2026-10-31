// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end flat expert: x relay primary (NCRISC). Its source is the helpers' tilized super-blocks (MT row tiles x
// SE_SBT K tiles, row tile major) in its landing rings (x read once by the helper relays: super-block b from helper
// b % SE_XNH, se13_xhelp.cpp): each of a super-block's x blocks ([MT x KBLK] tiles, row tile major in the gate/up
// cores' x ring) goes out as MT unicast pieces into the rectangle's snake head(s), once the block's ring slot is free
// there (the minimum of the heads' freed words); the gate/up cores forward it along the snake (se5_recv.cpp /
// se5_xfwd.cpp). The heads' XARR move by a batch's pieces once its writes are acknowledged.
// CT: 0 SB_CB, 1 MT, 2 TILE_BYTES, 3 X_SLOTS, 4 XARR_SEM, 5 WORD_SEM, 6 KBLK, 7 NUM_E, 8 NSB
// RT: 0 x ring address, 1 / 2 (unused), 3 heads, 4 freed words address, 5 heads whose words gate a slot,
//     6 (unused), 7 XARR semaphore id, 8 STRIDE, 9 OFF (this relay sends super-blocks OFF, OFF + STRIDE, ... in stream
//     order; a block's ring slot follows its global index), 10 row tiles to send: m0 | m1 << 8 (0: all MT), 11-13
//     (unused), 14 helper 0 xy, 15 landing rings address, 16 landing slots, 17.. helpers 1.. xy, then the se_dyn.hpp
//     args; RT 2 names the heads' list's first arg
#include <stdint.h>
#ifndef SE_SBT
#define SE_SBT 32  // K tiles per super-block (the row-major chunk width / 32 columns)
#endif
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"

void kernel_main() {
    constexpr uint32_t sb_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mt = get_compile_time_arg_val(1);
    constexpr uint32_t tb = get_compile_time_arg_val(2);
    constexpr uint32_t x_slots = get_compile_time_arg_val(3);
    constexpr uint32_t word_sem = get_compile_time_arg_val(5);
    constexpr uint32_t kblk = get_compile_time_arg_val(6);
    constexpr uint32_t per_sb = SE_SBT / kblk;
    constexpr uint32_t sb_tiles = mt * SE_SBT;
    static_assert(SE_SBT % kblk == 0);
    constexpr uint32_t blk_bytes = mt * kblk * tb, piece = kblk * tb;
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t a1 = get_arg_val<uint32_t>(2), dests = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
    const uint32_t n_cores = get_arg_val<uint32_t>(5);
    // the freed words restart every launch: zeroed here (the gate/up cores only bump them after consuming x this
    // relay sent in this launch), so no host write is needed between launches
    for (uint32_t i = 0; i < n_cores; ++i) {
        freed[i] = 0;
    }
    const uint32_t rows_rt = get_arg_val<uint32_t>(10);
    const uint32_t m0 = rows_rt ? (rows_rt & 0xFF) : 0, m1 = rows_rt ? (rows_rt >> 8) : mt;
    // Dynamic counts: this relay's super-blocks follow from the active experts' sub-blocks (RT 14.. are the
    // se_dyn.hpp args, CT 7 NUM_E, CB 7's upper half this RISC's scratch)
    constexpr uint32_t num_e = get_compile_time_arg_val(7), nsb = get_compile_time_arg_val(8);
    constexpr uint32_t dyn0 = 16 + SE_XNH;
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, dyn0, get_write_ptr(tt::CBIndex::c_7) + 2 * SE_DYN_HALF, mt * 32);
    const uint32_t tot_sb = dyn.num_v * nsb, st_ = get_arg_val<uint32_t>(8), of_ = get_arg_val<uint32_t>(9);
    const uint32_t num_sb = tot_sb > of_ ? (tot_sb - of_ + st_ - 1) / st_ : 0;
    volatile tt_l1_ptr uint32_t* word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(word_sem));
    const uint32_t xarr_addr = get_semaphore(get_arg_val<uint32_t>(7));
    const uint32_t stride = get_arg_val<uint32_t>(8), soff = get_arg_val<uint32_t>(9);
    // global index of this relay's k-th block
    auto gidx = [&](uint32_t k) { return ((k / per_sb) * stride + soff) * per_sb + k % per_sb; };
    auto min_freed = [&]() {
        uint32_t lo = freed[0];
        for (uint32_t i = 1; i < n_cores; ++i) {
            lo = freed[i] < lo ? freed[i] : lo;
        }
        return lo;
    };
    uint32_t sent = 0;
    const uint32_t x_heads = dests;                                             // (one head per snake)
    auto x_head = [&](uint32_t hd) { return get_arg_val<uint32_t>(a1 + hd); };  // (RT 2: the list's first arg)
    constexpr uint32_t XMC_CHAIN_DEPTH = 4;  // batches of head writes in flight (transaction ids 1..DEPTH)
    // Batch b's head writes carry transaction id 1 + b % DEPTH; once they are acknowledged the heads' XARR move by
    // that batch's pieces, in batch order (a head counts a contiguous prefix).
    uint32_t cb_iss = 0, cb_done = 0, cb_n[XMC_CHAIN_DEPTH];
    auto chain_retire = [&](uint32_t keep) {  // retire acknowledged batches; block while more than `keep` are out
        // (the write command buffer drained first: an id's outstanding count only covers requests the NIU has taken -
        // right after issuing, the last write can still sit in the command buffer and the id reads "done" early)
        while (cb_done < cb_iss && !noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        }
        while (cb_done < cb_iss && (cb_iss - cb_done > keep || ncrisc_noc_nonposted_write_with_transaction_id_flushed(
                                                                   noc_index, 1 + cb_done % XMC_CHAIN_DEPTH))) {
            while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + cb_done % XMC_CHAIN_DEPTH)) {
            }
            for (uint32_t hd = 0; hd < x_heads; ++hd) {
                const uint32_t hxy = x_head(hd);
                noc_semaphore_inc(
                    get_noc_addr(hxy >> 16, hxy & 0xFFFF, xarr_addr),
                    cb_n[cb_done % XMC_CHAIN_DEPTH] * XMC_CHAIN_PIECES);
            }
            ++cb_done;
        }
    };
    for (uint32_t b = 0; b < num_sb; ++b) {
        // Super-block b was read and tilized by helper h = b % NH (se13_xhelp.cpp) into landing ring h here (RT 14
        // helper 0 xy, 17.. helpers 1.. xy, 15 landing address (ring h follows ring h - 1), 16 landing slots; DATA sem
        // help_sem[h] here counts arrivals, the CREDIT sem XMC_CREDIT_SEM on the helper counts freed slots).
        constexpr uint32_t hs = SE_XNH;
        const uint32_t hh = b % hs, hb = b / hs;
        constexpr uint32_t help_sem[4] = {4, 7, 8, 10};
        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(help_sem[hh])) < hb + 1) {
            invalidate_l1_cache();
            chain_retire(XMC_CHAIN_DEPTH);  // (waiting: count acknowledged batches into the heads' XARR)
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(help_sem[hh])), hb + 1);
        const uint32_t src = get_arg_val<uint32_t>(15) +
                             (hh * get_arg_val<uint32_t>(16) + hb % get_arg_val<uint32_t>(16)) * sb_tiles * tb;
        uint32_t i = 0;
        while (i < per_sb) {
            invalidate_l1_cache();
            const uint32_t lim = min_freed() + x_slots;
            uint32_t n = 0;
            while (i + n < per_sb && gidx(sent + n) < lim) {
                ++n;
            }
            if (i + n < per_sb) {  // wait for the whole super-block's slots: one longer linked burst
                n = 0;
            }
            if (!n) {
                // Batches in flight are only counted into the heads' XARR on retirement: every wait retires the
                // acknowledged ones, else a compute that needs the whole ring (row passes: all Ht / KBLK blocks of a
                // sub-block resident, GLM-5.3 24 of 24) never sees the last batch and never frees one (deadlock). The
                // wait is for the same slot the batch needs (with XMC_WHOLE_SB the super-block's last block - waiting
                // on the first one only could spin here without retiring: codex review) and retires at least once
                do {
                    invalidate_l1_cache();
                    chain_retire(XMC_CHAIN_DEPTH);
                } while (gidx(sent + (per_sb - 1 - i)) >= min_freed() + x_slots);
                continue;
            }
            // the previous counter write has left L1 before its source word is reused
            while (!ncrisc_noc_nonposted_writes_sent(noc_index)) {
            }
            *word = sent + n;
            // the batch's pieces into every snake head (unicast, under the batch's transaction id); the heads forward
            // them along the snakes, se5_recv.cpp
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t dst = ring + (gidx(sent + k) % x_slots) * blk_bytes;
                for (uint32_t m = m0; m < m1; ++m) {
                    if (k == 0 && m == m0) {
                        chain_retire(XMC_CHAIN_DEPTH - 1);  // (room for this batch's id)
                    }
                    for (uint32_t hd = 0; hd < x_heads; ++hd) {
                        const uint32_t hxy = x_head(hd);
                        noc_async_write_one_packet_with_trid(
                            src + (m * SE_SBT + (i + k) * kblk) * tb,
                            get_noc_addr(hxy >> 16, hxy & 0xFFFF, dst + m * piece),
                            piece,
                            1 + cb_iss % XMC_CHAIN_DEPTH);
                    }
                }
            }
            cb_n[cb_iss % XMC_CHAIN_DEPTH] = n;
            ++cb_iss;
            chain_retire(XMC_CHAIN_DEPTH);
            sent += n;
            i += n;
            noc_async_writes_flushed();
        }
        const uint32_t hxy = get_arg_val<uint32_t>(hh ? 16 + hh : 14);
        noc_semaphore_inc(get_noc_addr(hxy >> 16, hxy & 0xFFFF, get_semaphore(XMC_CREDIT_SEM)), 1);  // slot free
    }
    chain_retire(0);
    noc_async_write_set_trid(0);
    noc_async_write_barrier();
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
