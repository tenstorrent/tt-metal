// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Row-major bf16 y (SE_Y_RM, with SE_DYN): y is [rows, H] bf16 row major, one page (H * 2 bytes) per flat row. The
// down compute pack-untilizes each row tile of its [MT x PCD] block and pushes it as PCD pages (32 rows x PCD * 32
// bf16, row after row); the writer sends each token row's PCD * 64 bytes to y[row, col0 * 32 ..], skipping rows at or
// past the entry's token count. Up to ROWS row tiles (the out CB's capacity) are in flight, each on its own write
// transaction id (TRID0 + k); they retire in order once their id is flushed. A virtual expert (entry a, sub-block s)
// is complete after its row tiles that hold tokens, min(MT, ceil((cnt - s MT 32) / 32)): the count the compute uses.
// The caller resets the write transaction id to 0 before the kernel exits.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"

template <uint32_t out_cb, uint32_t pcd, uint32_t mt, uint32_t ht>
struct SeYRmWriter {
    static constexpr uint32_t seg = pcd * 64;           // one token row of this core's columns (bytes)
    static constexpr uint32_t tile_bytes = pcd * 2048;  // one row tile in the out CB
    static constexpr uint32_t rows = SE_Y_RM_ROWS;      // out CB capacity in row tiles (host: yrm_rows)
    static constexpr uint32_t trid0 = 8;
    // The y writes' unicast VC: not the default write VC (1), which the weight forwarding, the h chain and the read
    // requests share; a y write queued behind them there stalls this core's NIU, and through its L1 reads the down
    // matmul's unpacker (~+20% expert time at large M, test_flat_balanced_sweep.py)
#ifdef SE_YRM_VC
    static constexpr uint8_t yvc = SE_YRM_VC;
#else
    static constexpr uint8_t yvc = 2;
#endif
    static_assert(rows >= 1 && trid0 + rows <= 16);
    InterleavedAddrGen<true> yg;
    uint32_t col_off;
    uint32_t base = 0;  // the out CB's start (its read pointer before anything is consumed)
    // issue cursor (entry, sub-block, row tile) and the in-order retire cursor
    uint32_t ia = 0, is = 0, ir = 0, issued = 0;
    uint32_t da = 0, ds = 0, dr = 0, retired = 0;
    uint32_t done_a = 0;  // the entry of the virtual expert the last retire() completed

#ifdef SE_Y_L1
    // L1 hand-off (flat_routed_expert_with_program + MIMO_FL_Y_L1): no y in DRAM; each token row's segment goes
    // straight into a slot of the row's destination core (a combine port's injection ring, or an accumulator's local
    // ring), from a per-chip table (per 32-row tile one 256-byte page: (destination, header word) per row). A
    // destination numbers its rows in the order the writers produce them (all writers walk the same schedule, so they
    // agree), grants slots (cumulative grant[d] in this core's words, +gbase[d] at this call's start), and takes a
    // row once every writer's landed counter there covers it (+1 per row, sent when a row tile's writes are acked).
    // The column-0 writer also writes each row's 32-byte trailer (header word, arrival flags). Args at SE_Y_L1:
    // nd, slot stride, table address, words address (grant[32], gbase[32], table page, trailer stages), writer index,
    // 3 spare, then per destination: noc x, noc y, slot base, slots, landed counter array address.
    static constexpr uint32_t L1_MAXD = 32;
    uint32_t nd = 0, islot = 0, words = 0, widx = 0, tab_tile = 0xFFFFFFFF, tab_base = 0;
    uint32_t dx[L1_MAXD], dy[L1_MAXD], dbase[L1_MAXD], dq[L1_MAXD], dland[L1_MAXD];
    uint32_t cnt[L1_MAXD];               // rows issued per destination this call
    uint8_t tcnt[rows][L1_MAXD];         // rows per destination of each in-flight row tile
    volatile tt_l1_ptr uint32_t* grant;  // words[0..32): grants received (cumulative)
    volatile tt_l1_ptr uint32_t* gbase;  // words[32..64): grants consumed by earlier calls
#endif

    SeYRmWriter(uint32_t y_addr, uint32_t col0) :
        yg{.bank_base_address = y_addr, .page_size = ht * 64}, col_off(col0 * 64), base(get_read_ptr(out_cb)) {
#ifdef SE_Y_L1
        constexpr uint32_t A0 = SE_Y_L1;
        nd = get_arg_val<uint32_t>(A0 + 0);
        islot = get_arg_val<uint32_t>(A0 + 1);
        tab_base = get_arg_val<uint32_t>(A0 + 2);
        words = get_arg_val<uint32_t>(A0 + 3);
        widx = get_arg_val<uint32_t>(A0 + 4);
        for (uint32_t d = 0; d < nd; ++d) {
            dx[d] = get_arg_val<uint32_t>(A0 + 8 + 5 * d);
            dy[d] = get_arg_val<uint32_t>(A0 + 9 + 5 * d);
            dbase[d] = get_arg_val<uint32_t>(A0 + 10 + 5 * d);
            dq[d] = get_arg_val<uint32_t>(A0 + 11 + 5 * d);
            dland[d] = get_arg_val<uint32_t>(A0 + 12 + 5 * d);
            cnt[d] = 0;
        }
        grant = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words);
        gbase = grant + 32;
#endif
    }
#ifdef SE_Y_L1
    // after the last row: the grants this call consumed
    void finish() {
        for (uint32_t d = 0; d < nd; ++d) {
            gbase[d] = gbase[d] + cnt[d];
        }
    }
#endif

    static uint32_t rows_v(const SeDyn& dyn, uint32_t a, uint32_t s) {
        const uint32_t left = dyn.cnt[a] - s * mt * 32;
        return left >= mt * 32 ? mt : (left + 31) / 32;
    }

    // Issue the next row tile's writes if the compute has pushed it and a transaction id is free; true if issued.
    bool issue(const SeDyn& dyn) {
        if (issued - retired == rows || !cb_pages_available_at_front(out_cb, (issued - retired + 1) * pcd)) {
            return false;
        }
        const uint32_t src = base + (issued % rows) * tile_bytes;
        const uint32_t row_base = is * mt * 32 + ir * 32;
        const uint32_t left = dyn.cnt[ia] - row_base;
        const uint32_t n = left < 32 ? left : 32;
        const uint32_t row0 = dyn.off[ia] + row_base;
        const uint32_t trid = trid0 + issued % rows;
#if defined(SE_Y_L1)
        if (nd == 0) {  // (nothing to hand off to: the flat expert alone without y)
            (void)src, (void)row0;
            ++issued;
            for (uint32_t d = 0; d < L1_MAXD; ++d) {
                tcnt[(issued - 1) % rows][d] = 0;
            }
            goto advanced;
        }
        {
            // this tile's table page: (destination, header word) per row
            const uint32_t page = row0 / 32, tabl = words + 256;
            if (page != tab_tile) {
                const InterleavedAddrGen<true> tg = {.bank_base_address = tab_base, .page_size = 256};
                noc_async_read(get_noc_addr(page, tg), tabl, 256);
                noc_async_read_barrier();
                tab_tile = page;
            }
            const volatile tt_l1_ptr uint32_t* ent = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(tabl);
            uint8_t* tc = tcnt[issued % rows];
            for (uint32_t d = 0; d < nd; ++d) {
                tc[d] = 0;
            }
            for (uint32_t i = 0; i < n; ++i) {
                ++tc[ent[2 * i] & 0xFF];
            }
            invalidate_l1_cache();
            for (uint32_t d = 0; d < nd; ++d) {  // every slot this tile needs granted?
                if (tc[d] && grant[d] - gbase[d] < cnt[d] + tc[d]) {
                    return false;
                }
            }
            const uint32_t stage = words + 512 + (issued % rows) * 32 * 32;  // trailers in flight (column-0 writer)
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t d = ent[2 * i] & 0xFF, hdr = ent[2 * i + 1];
                const uint32_t slot = dbase[d] + (cnt[d] % dq[d]) * islot;
                ++cnt[d];
                noc_async_write_one_packet_with_trid(
                    src + i * seg,
                    get_noc_addr(dx[d], dy[d], slot + col_off),
                    seg,
                    trid,
                    write_cmd_buf,
                    noc_index,
                    yvc);
                if (col_off == 0) {
                    volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + i * 32);
                    tr[0] = hdr;
                    tr[1] = tr[2] = tr[4] = tr[5] = tr[6] = 0;
                    tr[3] = 1;
                    tr[7] = 1;
                    noc_async_write_one_packet_with_trid(
                        stage + i * 32,
                        get_noc_addr(dx[d], dy[d], slot + ht * 64),
                        32,
                        trid,
                        write_cmd_buf,
                        noc_index,
                        yvc);
                }
            }
            (void)row0;
        }
#elif defined(SE_YRM_SELF)  // (perf probe: y rows written onto themselves in this core's L1 - no DRAM / mesh traffic,
        // the writes still issue and retire)
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write_one_packet_with_trid(
                src + i * seg,
                get_noc_addr(my_x[noc_index], my_y[noc_index], src + i * seg),
                seg,
                trid,
                write_cmd_buf,
                noc_index,
                yvc);
        }
        (void)row0;
#elif !defined(SE_YRM_NOWRITE)  // (perf probe: no y writes)
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write_one_packet_with_trid(
                src + i * seg, get_noc_addr(row0 + i, yg, col_off), seg, trid, write_cmd_buf, noc_index, yvc);
        }
#else
        (void)src, (void)n, (void)row0, (void)trid;
#endif
        ++issued;
#if defined(SE_Y_L1)
    advanced:
#endif
        if (++ir == rows_v(dyn, ia, is)) {
            ir = 0;
            if (++is == dyn.subs[ia]) {
                is = 0;
                ++ia;
            }
        }
        return true;
    }

    // Retire the oldest issued row tile once its writes are flushed (frees its pages); returns the number of virtual
    // experts completed by it (0 or 1).
    uint32_t retire(const SeDyn& dyn) {
        if (retired == issued ||
            !ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid0 + retired % rows)) {
            return 0;
        }
        cb_pop_front(out_cb, pcd);
#ifdef SE_Y_L1
        {
            const uint8_t* tc = tcnt[retired % rows];
            for (uint32_t d = 0; d < nd; ++d) {
                if (tc[d]) {
                    noc_semaphore_inc(get_noc_addr(dx[d], dy[d], dland[d] + 4 * widx), tc[d]);
                }
            }
        }
#endif
        ++retired;
        if (++dr == rows_v(dyn, da, ds)) {
            dr = 0;
            done_a = da;
            if (++ds == dyn.subs[da]) {
                ds = 0;
                ++da;
            }
            return 1;
        }
        return 0;
    }
};
