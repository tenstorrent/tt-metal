// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flat spatial expert: a bank reader that also computes some down columns (BRISC, NOC0). One non-blocking loop does
//   * the reader's job (se_reader.cpp): its gate/up weight region, chunk by chunk, into the forwarder's CB,
//   * the down weight reads for its own down columns (se6_dw.cpp) into the down compute's in1 ring,
//   * a down-chain tail's h handling (se6_drecv.cpp with no successor): h(v) arrives from the chain predecessor in
//     H_PIECES pieces (HARR), goes to compute once complete, and consumed h is reported back (HSFREE, atomic delta),
//   * the output: [MT x PCD] bf16 tiles of each virtual expert to y in DRAM, then a done report to the coordinator.
// Reads of both weight streams share one read barrier per loop pass (a pass issues at most one chunk of each).
//
// CT: 0 GU_CB, 1 W_TILE_BYTES, 2 GU_SLOT_TILES, 3 GU_CHUNKS (all experts), 4 D_CB (in1), 5 D_SLOT_TILES, 6 D_BLOCKS
//     (all experts), 7 H_ALL_CB, 8 H_ALL_TILES, 9 H_TILE_BYTES, 10 H_PIECES, 11 OUT_CB, 12 OUT_TILES, 13 NUM_V,
//     14 HARR_SEM, 15 HSFREE_SEM, 16 HBUF, 17 MT, 18 PCD, 19 HT, 20 S (sub-blocks per expert)
// RT: 0 gu weight bank base, 1 gu bank id, 2 gu region offset, 3 down weight bank base, 4 down bank id, 5 down region
//     offset, 6 predecessor xy, 7 coordinator xy, 8 done words address (coordinator), 9 this core's done index,
//     10 y DRAM address, 11 first y tile column
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DYN
#include "se_dyn.hpp"
#ifdef SE_Y_RM
#include "se_yrm.hpp"
#if !defined(SE_DYN) || !defined(SE_E2E) || defined(SE_SMALL_T)
#error "SE_Y_RM needs SE_DYN + SE_E2E (and no SE_SMALL_T)"
#endif
#endif
#endif
#if defined(SE_DN_REG) && !defined(SE9_TRID)
#error "pinned down ring needs the trid reader (SE9_TRID)"
#endif

void kernel_main() {
    constexpr uint32_t gu_cb = get_compile_time_arg_val(0);
    constexpr uint32_t w_tile = get_compile_time_arg_val(1);
    constexpr uint32_t gu_slot = get_compile_time_arg_val(2);
    constexpr uint32_t gu_chunks_ct = get_compile_time_arg_val(3);
    constexpr uint32_t d_cb = get_compile_time_arg_val(4);
    constexpr uint32_t d_slot = get_compile_time_arg_val(5);
    constexpr uint32_t d_blocks_ct = get_compile_time_arg_val(6);
    constexpr uint32_t h_all_cb = get_compile_time_arg_val(7);
    constexpr uint32_t h_all_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t h_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t h_pieces = get_compile_time_arg_val(10);
    constexpr uint32_t out_cb = get_compile_time_arg_val(11);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t num_v_ct = get_compile_time_arg_val(13);
    constexpr uint32_t harr_sem_id = get_compile_time_arg_val(14);
    constexpr uint32_t hsfree_sem_id = get_compile_time_arg_val(15);
    constexpr uint32_t hbuf = get_compile_time_arg_val(16);
    constexpr uint32_t mt = get_compile_time_arg_val(17);
    constexpr uint32_t pcd = get_compile_time_arg_val(18);
    constexpr uint32_t ht = get_compile_time_arg_val(19);
    constexpr uint32_t sub = get_compile_time_arg_val(20);
    constexpr uint32_t gu_bytes = gu_slot * w_tile;
    constexpr uint32_t d_bytes = d_slot * w_tile;

    uint64_t gu_src =
        get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(2));
    uint64_t d_src =
        get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(3) + get_arg_val<uint32_t>(5));
    const uint32_t pxy = get_arg_val<uint32_t>(6), cxy = get_arg_val<uint32_t>(7);
    const uint64_t pred_free = get_noc_addr(pxy >> 16, pxy & 0xFFFF, get_semaphore(hsfree_sem_id));
    const uint64_t done_noc =
        get_noc_addr(cxy >> 16, cxy & 0xFFFF, get_arg_val<uint32_t>(8) + 4 * get_arg_val<uint32_t>(9));
#ifdef SE_E2E
    // y is the bfp8 tile buffer shaped like the dispatch buffer: virtual expert v = (e, s) writes row tiles
    // region_tile(e) + s * MT + r, only those below the expert's token count (RT: per expert region tile, count tiles)
    constexpr uint32_t out_page = 1088;
    const InterleavedAddrGenFast<true> y = {
        .bank_base_address = get_arg_val<uint32_t>(10), .page_size = out_page, .data_format = DataFormat::Bfp8_b};
    const uint32_t e2e_args = 12;
#else
    constexpr uint32_t out_page = 2048;
    const InterleavedAddrGenFast<true> y = {
        .bank_base_address = get_arg_val<uint32_t>(10), .page_size = 2048, .data_format = DataFormat::Float16_b};
#endif
    const uint32_t col0 = get_arg_val<uint32_t>(11);
#ifdef SE_Y_RM
    SeYRmWriter<out_cb, pcd, mt, ht> yw(get_arg_val<uint32_t>(10), col0);
#endif
    volatile tt_l1_ptr uint32_t* harr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(harr_sem_id));

#ifdef SE_DYN
    // Dynamic counts (with SE_E2E): only the active experts' weights and sub-blocks; RT 12.. are the se_dyn.hpp args
    // (CT 21 NUM_E), CB 7 this RISC's scratch; this core's down compute gets them in CB 6.
    constexpr uint32_t num_e = get_compile_time_arg_val(21);
    constexpr uint32_t gu_per_e = gu_chunks_ct / num_e, d_per_e = d_blocks_ct / num_e;
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, 12, get_write_ptr(tt::CBIndex::c_7), mt * 32);
#ifdef SE_SMALL_T
    // Small-M role split: every active expert is small, so this core only streams gate/up weights; the down cores
    // take its columns (its compute gets no experts, its chain predecessor forwards no h here).
    const bool small = dyn.small;
#else
    constexpr bool small = false;
#endif
    se_dyn_publish(dyn, tt::CBIndex::c_6, small);
    const uint32_t gu_chunks = dyn.n_load * gu_per_e;  // gate/up loads (a pinned expert's weights come once)
#ifdef SE_DN_REG
    const uint32_t d_blocks = small ? 0 : dyn.n_load * d_per_e, num_v = small ? 0 : dyn.num_v;  // pinned: per load
#else
    const uint32_t d_blocks = small ? 0 : dyn.n_act * d_per_e, num_v = small ? 0 : dyn.num_v;
#endif
    const uint64_t gu_base = gu_src, d_base = d_src;
    uint32_t y_a = 0, y_s = 0;
#else
    constexpr uint32_t gu_chunks = gu_chunks_ct, d_blocks = d_blocks_ct, num_v = num_v_ct;
#endif
    uint32_t gu_read = 0, d_read = 0;
#ifdef SE9_TRID
    uint32_t gu_iss = 0, d_iss = 0;
    const uint32_t gu_l1 = get_write_ptr(gu_cb), d_l1 = get_write_ptr(d_cb);  // slot 0 of each (nothing pushed yet)
#endif
    uint32_t h_pub = 0, h_cons = 0, h_rep = 0, out_done = 0;
    bool y_pending = false;
    while (gu_read < gu_chunks || d_read < d_blocks || out_done < num_v) {
        invalidate_l1_cache();
#ifdef SE9_TRID
        // weights, non-blocking: up to GU_DEPTH gate/up chunks and D_DEPTH down blocks in flight, each read tagged
        // with its own NoC transaction id and pushed once that id has no outstanding reads (in order per stream), so
        // the gate/up stream never waits behind a down block (CT 22 GU_SLOTS, 23 D_RING: the CBs' slot counts).
        {
            constexpr uint32_t gu_slots = get_compile_time_arg_val(22), d_ring = get_compile_time_arg_val(23);
            constexpr uint32_t gu_depth = gu_slots < 4 ? gu_slots : 4, d_depth = 2;
            while (gu_iss < gu_chunks && gu_iss - gu_read < gu_depth &&
                   cb_pages_reservable_at_back(gu_cb, (gu_iss - gu_read + 1) * gu_slot)) {
#ifdef SE_DYN
                const uint64_t src =
                    gu_base + (dyn.load_eid[gu_iss / gu_per_e] * gu_per_e + gu_iss % gu_per_e) * gu_bytes;
#else
                const uint64_t src = gu_src + gu_iss * gu_bytes;
#endif
                noc_async_read_set_trid(1 + gu_iss % gu_depth);
                noc_async_read(src, gu_l1 + (gu_iss % gu_slots) * gu_bytes, gu_bytes);
                ++gu_iss;
            }
            while (d_iss < d_blocks && d_iss - d_read < d_depth &&
                   cb_pages_reservable_at_back(d_cb, (d_iss - d_read + 1) * d_slot)) {
#ifdef SE_DYN
#ifdef SE_DN_REG
                static_assert(d_ring == SE_GU_NREG * d_per_e);
                const uint64_t src = d_base + (dyn.load_eid[d_iss / d_per_e] * d_per_e + d_iss % d_per_e) * d_bytes;
                const uint32_t d_dst = d_l1 + (dyn.region[d_iss / d_per_e] * d_per_e + d_iss % d_per_e) * d_bytes;
#else
                const uint64_t src = d_base + (dyn.eid[d_iss / d_per_e] * d_per_e + d_iss % d_per_e) * d_bytes;
                const uint32_t d_dst = d_l1 + (d_iss % d_ring) * d_bytes;
#endif
#else
                const uint32_t d_dst = d_l1 + (d_iss % d_ring) * d_bytes;
                const uint64_t src = d_src + d_iss * d_bytes;
#endif
                noc_async_read_set_trid(8 + d_iss % d_depth);
                noc_async_read(src, d_dst, d_bytes);
                ++d_iss;
            }
            while (gu_read < gu_iss && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + gu_read % gu_depth)) {
                cb_push_back(gu_cb, gu_slot);
                ++gu_read;
            }
            while (d_read < d_iss && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 8 + d_read % d_depth)) {
                cb_push_back(d_cb, d_slot);
                ++d_read;
            }
        }
#else
        // weights: at most one chunk of each stream per pass, one barrier for both
        const bool rd_gu = gu_read < gu_chunks && cb_pages_reservable_at_back(gu_cb, gu_slot);
#ifdef SE_GU_FIRST
        // gate/up weights first: a down block only when the forwarder's CB is full (they are needed much later)
        const bool rd_d = !rd_gu && d_read < d_blocks && cb_pages_reservable_at_back(d_cb, d_slot);
#else
        const bool rd_d = d_read < d_blocks && cb_pages_reservable_at_back(d_cb, d_slot);
#endif
#ifdef SE_DYN
        if (rd_gu) {
            gu_src = gu_base + (dyn.load_eid[gu_read / gu_per_e] * gu_per_e + gu_read % gu_per_e) * gu_bytes;
        }
        if (rd_d) {
            d_src = d_base + (dyn.eid[d_read / d_per_e] * d_per_e + d_read % d_per_e) * d_bytes;
        }
#endif
        if (rd_gu) {
            noc_async_read(gu_src, get_write_ptr(gu_cb), gu_bytes);
            gu_src += gu_bytes;
        }
        if (rd_d) {
#ifndef SE9_SKIP_DW  // perf experiment only: down weights not read (garbage)
            noc_async_read(d_src, get_write_ptr(d_cb), d_bytes);
#endif
            d_src += d_bytes;
        }
        if (rd_gu || rd_d) {
            noc_async_read_barrier();
            if (rd_gu) {
                cb_push_back(gu_cb, gu_slot);
                ++gu_read;
            }
            if (rd_d) {
                cb_push_back(d_cb, d_slot);
                ++d_read;
            }
        }
#endif
        // h from the chain predecessor
        if (h_pub < num_v && *harr >= (h_pub + 1) * h_pieces && h_pub - h_cons < hbuf) {
            cb_push_back(h_all_cb, h_all_tiles);
            ++h_pub;
        }
        if (h_cons < h_pub && cb_pages_reservable_at_back(h_all_cb, (hbuf - (h_pub - h_cons) + 1) * h_all_tiles)) {
            ++h_cons;
        }
        if (h_cons > h_rep) {
            noc_semaphore_inc(pred_free, h_cons - h_rep);
            h_rep = h_cons;
        }
        // output
#ifdef SE_Y_RM
        yw.issue(dyn);
        if (yw.retire(dyn)) {
            noc_semaphore_inc(done_noc, 1);
            ++out_done;
        }
#else
        if (!y_pending && out_done < num_v && cb_pages_available_at_front(out_cb, out_tiles)) {
            const uint32_t src = get_read_ptr(out_cb);
            for (uint32_t r = 0; r < mt; ++r) {
#if defined(SE_DYN)
                if (y_s * mt * 32 + r * 32 >= dyn.cnt[y_a]) {
                    continue;
                }
                const uint32_t trow = dyn.off[y_a] / 32 + y_s * mt + r;
#elif defined(SE_E2E)
                const uint32_t e = out_done / sub, s_ = out_done % sub;
                if (s_ * mt + r >= get_arg_val<uint32_t>(e2e_args + 2 * e + 1)) {
                    continue;
                }
                const uint32_t trow = get_arg_val<uint32_t>(e2e_args + 2 * e) + s_ * mt + r;
#else
                const uint32_t trow = out_done * mt + r;
#endif
                for (uint32_t c = 0; c < pcd; ++c) {
                    noc_async_write_tile(trow * ht + col0 + c, y, src + (r * pcd + c) * out_page);
                }
            }
            y_pending = true;
        }
        if (y_pending && ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            cb_pop_front(out_cb, out_tiles);
            noc_semaphore_inc(done_noc, 1);
            ++out_done;
            y_pending = false;
#ifdef SE_DYN
            if (++y_s == dyn.subs[y_a]) {
                y_s = 0;
                ++y_a;
            }
#endif
        }
#endif
    }
#ifdef SE9_TRID
    noc_async_read_set_trid(0);
#endif
#ifdef SE_Y_RM
    noc_async_write_set_trid(0);  // y rows went out on write transaction ids: leave the packet tag at 0
#endif
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
