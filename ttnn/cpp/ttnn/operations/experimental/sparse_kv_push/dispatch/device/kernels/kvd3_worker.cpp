// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_kv_dispatch v3 worker W(f, l, k) (BRISC): takes the packets the neighbour in direction f sends this worker on
// link l (they move in f ^ 1) out of its per-class landing pools, and for each row:
//   - writes it to its owner-major page of `output` if this chip needs it (DRAM);
//   - copies it, with its onward entry, into the open staging slot of each out port (continue in f ^ 1; on a first leg
//     the turns into the other axis), class c + 1, in this core's L1.
// A landing slot is taken only if every copy fits its (port, class + 1) staging (the open slot plus the full ones the
// copies make, next to those still waiting for TX, at most `stage`); else that class is skipped this round, never
// waited on: classes are served independently, deepest first (tools/kvd3_model.py). After a slot: one write barrier,
// then the filled staging slots are published to their TX (pub word, class word) and the landing slot's credit is owed
// to TX(f, l), which forwards it to the neighbour. END(c) at a pool head: publish the class-c + 1 partial slots, then
// tell every TX of this link that this producer is done with class c + 1. Done once END(c) came for every class and TX
// has pulled every published slot; then TX(f, l) is told.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#include "api/debug/assert.h"
#include "kvd3_common.hpp"
#ifdef KVD_COMPACT
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"
#endif

#include <cstdint>

using namespace kvd;
using namespace kvd3;

namespace {
constexpr uint8_t kXRing[] = {KVD_XRING};
constexpr uint8_t kYRing[] = {KVD_YRING};
}  // namespace

#ifdef KVD_FUSED_PLAN
// the fused plan (kvd4): this kernel runs the chip's dispatch-only plan first (sparse_push_plan.cpp, PLAN_FUSED)
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/plan/device/kernels/sparse_push_plan.cpp"
#endif

#ifdef KVD_SEQ
// zero a landing slot's stamps: header word 3, END's trailer word (in a data packet an entry word, which could look
// like a stamp to a later END), and the stamp word of rows 0..n-1
FORCE_INLINE void seq_clear(uint32_t slot_l1, uint32_t rows_off, uint32_t row_bytes, uint32_t n) {
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1)[kHdrSeq] = 0;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1 + 48)[0] = 0;
    for (uint32_t i = 1; i <= n; ++i) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1 + rows_off + i * row_bytes - 16)[0] = 0;
    }
}
#endif

void kernel_main() {
#ifdef KVD_FUSED_PLAN
    plan_prologue();
#endif
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr bool WX = get_compile_time_arg_val(3) != 0;
    constexpr bool WY = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t P = get_compile_time_arg_val(6);
    constexpr uint32_t G = get_compile_time_arg_val(7);
    constexpr uint32_t B = get_compile_time_arg_val(8);
    constexpr uint32_t pool = get_compile_time_arg_val(9);
    constexpr uint32_t stage = get_compile_time_arg_val(10);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t stripe = get_compile_time_arg_val(12);
    constexpr uint32_t owner_slot_rows = get_compile_time_arg_val(13);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(14);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(15);
    constexpr uint32_t cb_pool = get_compile_time_arg_val(16);
    constexpr uint32_t idle_flush = get_compile_time_arg_val(17);  // idle spins before partial slots are published
    constexpr uint32_t lanes3 = get_compile_time_arg_val(18);      // 1, or 2: producers split by index parity
    constexpr uint32_t wsplit =
        get_compile_time_arg_val(19);                        // 1, or 2: this core's BRISC and NCRISC split the classes
    constexpr uint32_t wpar = get_compile_time_arg_val(20);  // the class parity this RISC serves (wsplit 2)
    constexpr uint32_t sem_wgo = get_compile_time_arg_val(21);  // parity 0 releases parity 1 after the chip barrier
    constexpr auto output_args = TensorAccessorArgs<22>();
#ifdef KVD_COMPACT
    // compact landing: sparse_push_plan's rank table (page 0 the owners' run bases, page 1 + b block b's word ranks and
    // bitmap words); a row lands at run_base[owner] + its rank within its owner's run (sparse_push_plan plan_page)
    constexpr auto rank_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr uint32_t rank_page_bytes = KVD_RANK_PAGE_WORDS * 4;
    constexpr uint32_t words_per_block = stripe / 32;
#endif
    // classes this RISC serves (lands, routes, signals done for c + 1, owes credits for): c % wsplit == wpar; the
    // staging of class c + 1 is written only by the RISC serving c (one writer a slot)
    auto own = [](uint32_t c) { return wsplit == 1 || (c % 2) == wpar; };
    constexpr uint32_t sbytes = slot_bytes(B, row_bytes);
    constexpr uint32_t rows_off = slot_rows_offset(B);

    const uint32_t output_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t go_addr = get_common_arg_val<uint32_t>(9);
    const uint32_t barrier_addr = get_common_arg_val<uint32_t>(10);

    uint32_t a = 0;
    const uint32_t present = get_arg_val<uint32_t>(a++);
    const uint32_t bar_x = get_arg_val<uint32_t>(a++);
    const uint32_t bar_y = get_arg_val<uint32_t>(a++);
    const uint32_t ctrl = get_write_ptr(cb_ctrl);
    auto* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl);
    auto* wgo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_wgo));
#ifdef KVD_ACTIVE_FROM_BLOCK
    // (the stage CB: written by this core only after the gate; each RISC reads the header into its own 64 B)
    if (skip_chunk(get_common_arg_val<uint32_t>(12), align64(get_write_ptr(cb_stage)) + 64 * wpar)) {
        return;
    }
#endif
    if (wpar == 0) {  // one barrier member a core
#ifdef KVD_ABL_ZERO_LOOP
        for (uint32_t i = 0; i < kCtrlWords; ++i) {
            cw[i] = 0;
        }
#else
        zero_ctrl(ctrl);
#endif
#ifdef KVD_SEQ
        {  // no stale stamp in the landing pool (before the barrier: nothing lands before every chip passed it)
            const uint32_t pb = align64(get_write_ptr(cb_pool));
            for (uint32_t c = 0; c < G; ++c) {
                for (uint32_t s = 0; s < pool_depth(c, pool); ++s) {
                    seq_clear(pb + pool_offset(c, s, pool, sbytes), rows_off, row_bytes, B);
                }
            }
        }
#endif
        noc_semaphore_inc(get_noc_addr(bar_x, bar_y, barrier_addr), 1);
        noc_async_atomic_barrier();
        auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_addr);
        noc_semaphore_wait_min(go, 1);
        noc_semaphore_set(go, 0);
        if (wsplit == 2) {
            *wgo = 1;
        }
    } else {
        while (*wgo == 0) {
        }
    }
    if (!present) {
        return;
    }
    const uint32_t f = get_arg_val<uint32_t>(a++);
    const uint32_t l = get_arg_val<uint32_t>(a++);
    const uint32_t k = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t pidx = get_arg_val<uint32_t>(a++);
    const uint32_t lane3 = lanes3 == 2 ? (pidx & 1) : 0;     // this worker's producer index on the TX cores
    const uint32_t tx_present = get_arg_val<uint32_t>(a++);  // bit td: TX(td, l) exists
    uint32_t tx_x[4], tx_y[4];
    for (uint32_t td = 0; td < 4; ++td) {
        tx_x[td] = get_arg_val<uint32_t>(a++);
        tx_y[td] = get_arg_val<uint32_t>(a++);
    }
    uint32_t rd_x[4], rd_y[4];  // this worker's lane reader of TX(td, l): the TX core (lane 0) or its helper (lane 1)
    for (uint32_t td = 0; td < 4; ++td) {
        rd_x[td] = get_arg_val<uint32_t>(a++);
        rd_y[td] = get_arg_val<uint32_t>(a++);
    }
    (void)l;

    const auto output = TensorAccessor(output_args, output_addr, row_bytes);
    const uint32_t stage_base = align64(get_write_ptr(cb_stage));
    const uint32_t pool_base = align64(get_write_ptr(cb_pool));
    const Routing<R, C, WX, WY> rt{kXRing, kYRing};
    const uint32_t my_r = me / C, my_c = me % C;
    const uint32_t mv = f ^ 1;  // the packets here move in f ^ 1
    const bool mv_x = mv < 2;
    const uint32_t me_bit = 1u << me;
#ifdef KVD_COMPACT
    const auto rank_table = TensorAccessor(rank_args, get_common_arg_val<uint32_t>(13), rank_page_bytes);
    // this RISC's region of the rank CB: the run bases, then one rank page a row of a landing slot
    const uint32_t rank_l1 = align64(get_write_ptr(KVD_CB_RANK)) + wpar * (1 + (B > 8 ? B : 8)) * rank_page_bytes;
    noc_async_read(rank_table.get_noc_addr(0), rank_l1, rank_page_bytes);
    noc_async_read_barrier();
    const volatile tt_l1_ptr uint32_t* run_bases = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rank_l1);
#endif

    uint32_t processed[G], pub[4][G], openn[4][G];
    bool end_in[G];
    for (uint32_t c = 0; c < G; ++c) {
        processed[c] = 0;
        end_in[c] = false;
        for (uint32_t td = 0; td < 4; ++td) {
            pub[td][c] = 0;
            openn[td][c] = 0;
        }
    }
    auto open_slot = [&](uint32_t td, uint32_t c) {
        return stage_base + stage_offset(td, c, pub[td][c] % stage, G, stage, sbytes);
    };
    auto publish_signal = [&](uint32_t td, uint32_t c, uint32_t n) {
        noc_semaphore_inc(get_noc_addr(rd_x[td], rd_y[td], ctrl + tx_pub(pidx, c) * 4), n);
        noc_semaphore_inc(get_noc_addr(rd_x[td], rd_y[td], ctrl + tx_cls_pub(lane3, c) * 4), n);
        noc_semaphore_inc(get_noc_addr(rd_x[td], rd_y[td], ctrl + tx_pub_total(lane3) * 4), n);
    };
    // close the open slot of (td, c) (its rows and entries must already be visible)
    auto close_slot = [&](uint32_t td, uint32_t c) {
        auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(open_slot(td, c));
        h[kHdrEntries] = openn[td][c];
        h[kHdrClass] = c;
        h[kHdrEnd] = 0;
        openn[td][c] = 0;
        ++pub[td][c];
    };
    auto publish_partials = [&](uint32_t c_only) {  // c_only = G: every class
        bool any = false;
        for (uint32_t td = 0; td < 4; ++td) {
            for (uint32_t c = 1; c < G; ++c) {
                if (own(c - 1) && (c_only == G || c == c_only) && openn[td][c] != 0) {
                    close_slot(td, c);
                    publish_signal(td, c, 1);
                    any = true;
                }
            }
        }
        return any;
    };
    auto owe_credit = [&](uint32_t c) {
        noc_semaphore_inc(get_noc_addr(tx_x[f], tx_y[f], ctrl + tx_owed(k, c) * 4), 1);
#ifdef KVD_ABL_PIGGY
        noc_semaphore_inc(get_noc_addr(tx_x[f], tx_y[f], ctrl + tx_owed_k(k) * 4), 1);
#endif
        noc_semaphore_inc(get_noc_addr(tx_x[f], tx_y[f], ctrl + kTxOwedTotal * 4), 1);
    };

    // a row's onward copies: up to 3 (out port, hops)
    struct Out {
        uint32_t td, hops;
    };
#ifdef KVD_ABL_W_AGE
    // a partial slot older than KVD_ABL_W_AGE cycles leaves without waiting for B rows or END (thin deep classes)
    uint32_t opened[4][G];
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    auto age_out = [&]() {  // (every open slot's rows are visible: each processed slot ends with a barrier)
        const uint32_t now = wall();
        for (uint32_t td = 0; td < 4; ++td) {
            for (uint32_t c = 1; c < G; ++c) {
                if (own(c - 1) && openn[td][c] != 0 && now - opened[td][c] > KVD_ABL_W_AGE) {
                    close_slot(td, c);
                    publish_signal(td, c, 1);
                }
            }
        }
    };
#endif
    uint32_t idle = 0;
    uint32_t flush_seen[4] = {0, 0, 0, 0};
    (void)flush_seen;
    uint32_t n_slots = 0, n_rows = 0, n_blocked = 0, n_end_rows = 0, n_stamp_wait = 0;
    uint32_t n_cut = 0, n_cut_local = 0, n_cut_full = 0;  // (KVD_STATS: cut-through candidates)
    (void)n_cut;
    (void)n_cut_local;
    (void)n_cut_full;
    const uint32_t t0 = stats_clock();
    uint32_t cy_work = 0, cy_cap = 0, cy_rows = 0, cy_wbar = 0, cy_sig = 0, n_writes = 0;
    while (true) {
        bool progress = false;
        for (uint32_t cj = 0; cj < G; ++cj) {
#ifdef KVD_ABL_W_SHALLOW_FIRST
            const uint32_t c = cj;  // (any order is deadlock-free while a blocked class is skipped, never waited on)
#else
            const uint32_t c = G - 1 - cj;
#endif
            if (!own(c) || cw[w_arrived(c)] == processed[c]) {
                continue;
            }
            const uint32_t tw = stats_clock();
            const uint32_t slot_l1 = pool_base + pool_offset(c, processed[c] % pool_depth(c, pool), pool, sbytes);
            auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1);
#ifdef KVD_SEQ
            {
                const uint32_t want = kSeqMagic | (processed[c] + 1);
                bool waited = false;
                while (h[kHdrSeq] != want) {  // the header is this packet's
                    waited = true;
                }
                auto* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    slot_l1 + seq_trailer(B, row_bytes, h[kHdrEntries], h[kHdrEnd] != 0));
                while (tr[0] != want) {  // the whole packet has landed
                    waited = true;
                }
                n_stamp_wait += waited;
            }
#endif
#ifdef KVD_ABL_PIGGY
            // credits the neighbour piggybacked for TX(f, l) of this chip (its packets to the neighbour's workers):
            // taken once per slot (a blocked slot is seen again: clear the count once applied)
            if (h[kHdrCredits] != 0) {
                auto* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1 + credit_area_offset(B));
                for (uint32_t j = 0; j < h[kHdrCredits]; ++j) {
                    const uint32_t v = w[j];
                    noc_semaphore_inc(
                        get_noc_addr(tx_x[f], tx_y[f], ctrl + tx_credit(v & 7, (v >> 3) & 15) * 4), v >> 7);
                }
                h[kHdrCredits] = 0;
            }
#endif
            if (h[kHdrEnd]) {
                if (c + 1 < G) {
                    for (uint32_t td = 0; td < 4; ++td) {
                        n_end_rows += openn[td][c + 1];  // rows that leave only because END(c) closed their slot
                    }
                    publish_partials(c + 1);  // (rows already visible: every slot ends with a barrier)
                    noc_async_atomic_barrier();
                    for (uint32_t td = 0; td < 4; ++td) {
                        if (tx_present >> td & 1) {
                            noc_semaphore_inc(get_noc_addr(rd_x[td], rd_y[td], ctrl + tx_done(lane3, c + 1) * 4), 1);
                        }
                    }
                }
                end_in[c] = true;
#ifdef KVD_SEQ
                seq_clear(slot_l1, rows_off, row_bytes, 0);
#endif
                ++processed[c];
                owe_credit(c);
                progress = true;
                cy_work += stats_clock() - tw;
                continue;
            }
            const uint32_t n = h[kHdrEntries];
            ASSERT(n <= B);  // (producers close a slot at B entries; the arrays below hold B)
            auto* ents = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1 + kSlotHeaderBytes);
            // capacity: copies a port gets from this slot (each entry decoded once: the row loop reuses it)
            uint32_t need[4] = {0, 0, 0, 0};
            uint32_t e_w0[B], e_m[B], e_sp[B];
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t w0 = ents[2 * i], m = ents[2 * i + 1];
                const uint32_t g = entry_row(w0), hops = entry_hops(w0), order = entry_order(w0);
                e_w0[i] = w0;
                e_m[i] = m;
                e_sp[i] = 0;
                if (hops > 1) {
                    ++need[mv];
                }
                if (mv_x == (order == 0)) {
                    const uint32_t sp = rt.turn(!mv_x, my_r, my_c, m, hbit(g, kTieBit));
                    e_sp[i] = sp;
                    const uint32_t tp = mv_x ? 2u : 0u;
                    need[tp] += span_fwd(sp) ? 1 : 0;
                    need[tp + 1] += span_bwd(sp) ? 1 : 0;
                }
            }
            bool fits = true;
            for (uint32_t td = 0; td < 4 && fits; ++td) {
                if (need[td] == 0) {
                    continue;
                }
                if (c + 1 >= G) {
                    fits = false;  // (cannot happen: G bounds every generation + 1; the host sizes it from the span
                                   // tables)
                    break;
                }
                const uint32_t waiting = pub[td][c + 1] - cw[w_freed(td, c + 1)];
                const uint32_t total = openn[td][c + 1] + need[td];
                const uint32_t slots_needed = waiting + total / B + (total % B ? 1 : 0);
                fits = slots_needed <= stage;
            }
            if (!fits) {
                ++n_blocked;
                cy_work += stats_clock() - tw;
                continue;  // skip this class this round; never wait on it
            }
#ifdef KVD_STATS
            {  // cut-through candidates: every row continues straight (one copy, port mv), none ends or turns here
                bool straight = n != 0;
                uint32_t local = 0;
                for (uint32_t i = 0; i < n && straight; ++i) {
                    const uint32_t hops = entry_hops(e_w0[i]);
                    straight = hops > 1 && e_sp[i] == 0;
                    local += (e_m[i] & me_bit) ? 1 : 0;
                }
                n_cut += straight ? 1 : 0;
                n_cut_local += (straight && local == 0) ? 1 : 0;
                n_cut_full += (straight && n == B) ? 1 : 0;
            }
#endif
            const uint32_t trows = stats_clock();
            cy_cap += trows - tw;
#ifdef KVD_COMPACT
            // the rank pages of the rows this chip needs: issued now, waited for after the onward copies (below), so
            // the DRAM round trip overlaps them
            // (one read a block: a slot's rows mostly come from a few blocks; e_pg[i]: row i's page in the CB)
            uint32_t e_pg[B];
            {
                uint32_t last_blk = ~0u, pages = 0;
                for (uint32_t i = 0; i < n; ++i) {
                    if (e_m[i] & me_bit) {
                        const uint32_t blk = entry_row(e_w0[i]) / stripe;
                        if (blk != last_blk) {
                            noc_async_read(
                                rank_table.get_noc_addr(1 + blk),
                                rank_l1 + (1 + pages) * rank_page_bytes,
                                rank_page_bytes);
                            last_blk = blk;
                            ++pages;
                        }
                        e_pg[i] = pages;  // (1-based: page 0 holds the run bases)
                    }
                }
            }
#endif
            // the rows
            uint32_t to_publish[4] = {0, 0, 0, 0};
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t w0 = e_w0[i], m = e_m[i];
                const uint32_t g = entry_row(w0), hops = entry_hops(w0), gen = entry_gen(w0), order = entry_order(w0);
                const uint32_t src = slot_l1 + rows_off + i * row_bytes;
#ifndef KVD_ABL_W_NOGATHER
#ifndef KVD_COMPACT  // (compact: after the onward copies, once the rank pages are in)
                if (m & me_bit) {
                    const uint32_t owner = (g / stripe) % N;
                    noc_async_write(
                        src,
                        output.get_noc_addr(owner * owner_slot_rows + (g / stripe / N) * stripe + g % stripe),
                        row_bytes);
                }
#endif
#endif
                Out outs[3];
                uint32_t no = 0;
                if (hops > 1) {
                    outs[no++] = {mv, hops - 1};
                }
                if (mv_x == (order == 0)) {
                    const uint32_t sp = e_sp[i];
                    const uint32_t tp = mv_x ? 2u : 0u;
                    if (span_fwd(sp)) {
                        outs[no++] = {tp, span_fwd(sp)};
                    }
                    if (span_bwd(sp)) {
                        outs[no++] = {tp + 1, span_bwd(sp)};
                    }
                }
                for (uint32_t o = 0; o < no; ++o) {
                    const uint32_t td = outs[o].td, cc = c + 1;
                    const uint32_t dst = open_slot(td, cc);
                    const uint32_t j = openn[td][cc];
#ifdef KVD_ABL_W_AGE
                    if (j == 0) {
                        opened[td][cc] = wall();
                    }
#endif
                    auto* de = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + kSlotHeaderBytes);
                    de[2 * j] = entry_word0(g, outs[o].hops, gen + 1, order);
                    de[2 * j + 1] = m;
#ifndef KVD_ABL_W_NOCOPY
                    noc_async_write(src, get_noc_addr(dst + rows_off + j * row_bytes), row_bytes);
                    ++n_writes;
#endif
                    if (++openn[td][cc] == B) {
                        close_slot(td, cc);  // (published after the barrier below)
                        ++to_publish[td];
                    }
                }
                ++n_rows;
            }
#if defined(KVD_COMPACT) && !defined(KVD_ABL_W_NOGATHER)
            noc_async_read_barrier();  // the rank pages
            for (uint32_t i = 0; i < n; ++i) {
                if (e_m[i] & me_bit) {
                    const uint32_t g = entry_row(e_w0[i]);
                    const auto* rp =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rank_l1 + e_pg[i] * rank_page_bytes);
                    const uint32_t w = (g % stripe) / 32;
                    const uint32_t rank = rp[w] + __builtin_popcount(rp[words_per_block + w] & ((1u << (g % 32)) - 1u));
                    noc_async_write(
                        slot_l1 + rows_off + i * row_bytes,
                        output.get_noc_addr(run_bases[(g / stripe) % N] + rank),
                        row_bytes);
                }
            }
#endif
            const uint32_t tbar = stats_clock();
            cy_rows += tbar - trows;
            noc_async_write_barrier();  // copies and gather writes landed: the slots may be published, this one freed
            const uint32_t tsig = stats_clock();
            cy_wbar += tsig - tbar;
            for (uint32_t td = 0; td < 4; ++td) {
                if (to_publish[td]) {
                    publish_signal(td, c + 1, to_publish[td]);
                }
            }
#ifdef KVD_SEQ
            seq_clear(
                slot_l1, rows_off, row_bytes, n);  // (after the write barrier: every copy out of the slot is done)
#endif
            ++processed[c];
            owe_credit(c);
            cy_sig += stats_clock() - tsig;
#ifdef KVD_ABL_W_AGE
            age_out();
#endif
            ++n_slots;
            progress = true;
            cy_work += stats_clock() - tw;
        }
        if (progress) {
            idle = 0;
            continue;
        }
#ifdef KVD_ABL_W_AGE
        age_out();
#endif
        if (++idle == idle_flush) {
            publish_partials(G);  // nothing arrived for a while: let the open slots go
        }
#ifdef KVD_ABL_HUNGRY
        // a TX reader with nothing to send asked for the open slots of its port: publish them now
        for (uint32_t td = 0; td < 4; ++td) {
            if (cw[w_flush_req(td)] != flush_seen[td]) {
                flush_seen[td] = cw[w_flush_req(td)];
                for (uint32_t c = 1; c < G; ++c) {
                    if (own(c - 1) && openn[td][c] != 0) {
                        close_slot(td, c);
                        publish_signal(td, c, 1);
                    }
                }
            }
        }
#endif
        bool all_end = true;
        for (uint32_t c = 0; c < G; ++c) {
            all_end = all_end && (end_in[c] || !own(c));
        }
        if (all_end) {
            bool all_pulled = true;
            for (uint32_t td = 0; td < 4 && all_pulled; ++td) {
                for (uint32_t c = 1; c < G; ++c) {
                    if (!own(c - 1)) {
                        continue;
                    }
                    if (openn[td][c] != 0 || pub[td][c] != cw[w_freed(td, c)]) {
                        all_pulled = false;
                        break;
                    }
                }
            }
            if (all_pulled) {
                break;
            }
        }
    }
    noc_async_atomic_barrier();  // every credit owed is in TX(f, l)'s word before it hears this worker is done
    noc_semaphore_inc(get_noc_addr(tx_x[f], tx_y[f], ctrl + kTxWorkersDone * 4), 1);
    noc_async_atomic_barrier();
#ifdef KVD_STATS
    DPRINT(
        "KVD3W f={} k={} total={} work={} slots={} rows={} blocked={}\n",
        f,
        k,
        stats_clock() - t0,
        cy_work,
        n_slots,
        n_rows,
        n_blocked);
    DPRINT("KVD3E f={} k={} end_rows={} exit={}\n", f, k, n_end_rows, (stats_clock() - t0) / 135);
    DPRINT(
        "KVD3WX f={} slots={} straight={} straight_nolocal={} straight_full={}\n",
        f,
        n_slots,
        n_cut,
        n_cut_local,
        n_cut_full);
    DPRINT(
        "KVD3WC cap={} rows={} wbar={} sig={} writes={} stampw={}\n",
        cy_cap,
        cy_rows,
        cy_wbar,
        cy_sig,
        n_writes,
        n_stamp_wait);
#endif
    (void)t0;
    (void)cy_work;
    (void)n_slots;
    (void)n_rows;
    (void)n_blocked;
    (void)n_stamp_wait;
    (void)n_end_rows;
    (void)cy_cap;
    (void)cy_rows;
    (void)cy_wbar;
    (void)cy_sig;
    (void)n_writes;
}
