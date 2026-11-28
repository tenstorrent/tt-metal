// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_kv_dispatch port unpacker (NCRISC, port core (lane, d)): takes the packets the neighbour in direction d sends
// here (they move in direction d ^ 1) out of this core's landing ring. Every slot that has arrived is one batch, in
// this order (the ordering contract, KV_DISPATCH_2D_PLAN.md 4.3, checked by tools/kvd2d_model.py):
//   1. write each row to its owner-major page of `output`;
//   2. write the onward entries (continue on port d ^ 1, turns on the other axis' ports, all on this lane) into their
//      queues in DRAM scratch; one barrier covers 1 and 2;
//   3. raise each queue's tail on its consumer port core;
//   4. free the slots: raise credit_out, which this core's sender returns to the neighbour.
// It waits only on `arrived` and its own writes. After the slot holding END(GMAX) it subtracts what it processed from
// `arrived` (a persistent word, zero between calls) and raises unpack_done.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#include "kvd_common.hpp"

#include <cstdint>

using namespace kvd;

namespace {
constexpr uint8_t kXRing[] = {KVD_XRING};
constexpr uint8_t kYRing[] = {KVD_YRING};

// One output queue this unpacker produces (single producer): its staging page mirrors the queue's current DRAM page.
struct OutQueue {
    bool on = false;
    uint32_t base_page = 0;  // scratch page of the queue's first entry
    uint32_t tail_noc_x = 0, tail_noc_y = 0, tail_sem = 0;
    uint32_t written = 0, published = 0, flushed = 0;  // entries
    uint32_t stage_l1 = 0;
};
}  // namespace

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr bool WX = get_compile_time_arg_val(3) != 0;
    constexpr bool WY = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t stripe = get_compile_time_arg_val(5);
    constexpr uint32_t owner_slot_rows = get_compile_time_arg_val(6);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t B = get_compile_time_arg_val(8);
    constexpr uint32_t slots = get_compile_time_arg_val(9);
    constexpr uint32_t gmax = get_compile_time_arg_val(10);
    constexpr uint32_t cb_land = get_compile_time_arg_val(11);
    constexpr uint32_t cb_ustage = get_compile_time_arg_val(12);
    constexpr auto output_args = TensorAccessorArgs<13>();
    constexpr auto scratch_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr uint32_t sbytes = slot_bytes(B, row_bytes);
    constexpr uint32_t rows_off = slot_rows_offset(B);

    const uint32_t output_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t scratch_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t arrived_addr = get_common_arg_val<uint32_t>(9);

    uint32_t a = 0;
    const uint32_t has_neighbour = get_arg_val<uint32_t>(a++);
    const uint32_t d = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t unpack_done_sem = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t credit_out_sem = get_semaphore(get_arg_val<uint32_t>(a++));
    // the three output queues: continue, turn +, turn - (on, base page, consumer noc x, y, tail semaphore id)
    OutQueue q[3];
    for (uint32_t k = 0; k < 3; ++k) {
        q[k].on = get_arg_val<uint32_t>(a++) != 0;
        q[k].base_page = get_arg_val<uint32_t>(a++);
        q[k].tail_noc_x = get_arg_val<uint32_t>(a++);
        q[k].tail_noc_y = get_arg_val<uint32_t>(a++);
        q[k].tail_sem = get_semaphore(get_arg_val<uint32_t>(a++));
    }

    auto* unpack_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(unpack_done_sem);
#ifdef KVD_ABL_SYNTH
    if (has_neighbour) {  // the synthetic transport's one closing increment
        auto* arr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrived_addr);
        while (*arr < 1) {
        }
        noc_semaphore_inc(get_noc_addr(arrived_addr), 0u - 1u);
    }
#endif
#if defined(KVD_ABL_NO_TRAVERSE) || defined(KVD_ABL_SYNTH)
    (void)has_neighbour;
    if (true) {
#else
    if (!has_neighbour) {
#endif
        noc_semaphore_inc(get_noc_addr(unpack_done_sem), 1);
        noc_async_atomic_barrier();
        return;
    }
    (void)unpack_done;

    const auto output = TensorAccessor(output_args, output_addr, row_bytes);
    const auto scratch = TensorAccessor(scratch_args, scratch_addr, kScratchPageBytes);
    const uint32_t land_l1 = align64(get_write_ptr(cb_land));
    cb_reserve_back(cb_ustage, 1);
    const uint32_t ustage_l1 = align64(get_write_ptr(cb_ustage));
    for (uint32_t k = 0; k < 3; ++k) {
        q[k].stage_l1 = ustage_l1 + k * kScratchPageBytes;
    }

    const Routing<R, C, WX, WY> rt{kXRing, kYRing};
    const uint32_t my_r = me / C, my_c = me % C;
    const uint32_t move = d ^ 1;  // packets landing here move in direction d ^ 1
    const bool move_x = move < 2;
    auto* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrived_addr);

    auto append = [&](OutQueue& o, uint32_t w0, uint32_t w1) {
        auto* e =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(o.stage_l1 + (o.written % kEntriesPerPage) * kEntryBytes);
        e[0] = w0;
        e[1] = w1;
        ++o.written;
        if (o.written % kEntriesPerPage == 0) {  // a full page: write it out whole
            const uint32_t page = o.base_page + (o.written - 1) / kEntriesPerPage;
            const uint32_t from = (o.flushed % kEntriesPerPage) * kEntryBytes & ~63u;
            noc_async_write(o.stage_l1 + from, scratch.get_noc_addr(page) + from, kScratchPageBytes - from);
            noc_async_write_barrier();  // the staging page is reused next
            o.flushed = o.written;
        }
    };
    auto flush_partial = [&](OutQueue& o) {
        if (o.written == o.flushed) {
            return;
        }
        const uint32_t page = o.base_page + o.written / kEntriesPerPage;
        const uint32_t from = ((o.flushed % kEntriesPerPage) * kEntryBytes) & ~63u;
        const uint32_t to = ((o.written % kEntriesPerPage) * kEntryBytes + 63) & ~63u;
        noc_async_write(o.stage_l1 + from, scratch.get_noc_addr(page) + from, to - from);
        o.flushed = o.written;
    };

    // Every slot that has arrived is one batch: its rows and onward entries are written, one barrier covers both,
    // then the queue tails and the batch's credits are published.
    uint32_t processed = 0;
    bool ended = false;
    KVD_ZONE("KVD-U-unpack");
    auto now = stats_clock;
    uint32_t cy_wait = 0, cy_work = 0, n_batches = 0;
    const uint32_t t_start = now();
    while (!ended) {
        const uint32_t t0 = now();
        const uint32_t avail = *arrived - processed;
        if (avail == 0) {
            cy_wait += now() - t0;
            continue;
        }
        ++n_batches;
        KVD_ZONE("KVD-U-batch");
        // 1. rows to their owner-major pages
        for (uint32_t s_i = 0; s_i < avail; ++s_i) {
            const uint32_t slot_l1 = land_l1 + ((processed + s_i) % slots) * sbytes;
            const uint32_t entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1)[0];
            auto* ents = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1 + kSlotHeaderBytes);
            uint32_t row_k = 0;
            for (uint32_t i = 0; i < entries; ++i) {
                const uint32_t w0 = ents[2 * i];
                if (w0 & kEndBit) {
                    continue;
                }
                const uint32_t g = entry_row(w0);
                const uint32_t owner = (g / stripe) % N;
                const uint32_t page = owner * owner_slot_rows + (g / stripe / N) * stripe + g % stripe;
#ifndef KVD_ABL_NO_ROW_WRITE
                noc_async_write(slot_l1 + rows_off + row_k * row_bytes, output.get_noc_addr(page), row_bytes);
#else
                (void)page;
#endif
                ++row_k;
            }
        }
        // 2. onward entries
        for (uint32_t s_i = 0; s_i < avail; ++s_i) {
            const uint32_t slot_l1 = land_l1 + ((processed + s_i) % slots) * sbytes;
            const uint32_t entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1)[0];
            auto* ents = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1 + kSlotHeaderBytes);
            for (uint32_t i = 0; i < entries; ++i) {
                const uint32_t w0 = ents[2 * i], m = ents[2 * i + 1];
                if (w0 & kEndBit) {
                    if (entry_gen(w0) == gmax) {
                        ended = true;
                    }
                    for (uint32_t k = 0; k < 3; ++k) {
                        if (q[k].on) {
                            append(q[k], w0, 0);
                        }
                    }
                    continue;
                }
                const uint32_t g = entry_row(w0), h = entry_hops(w0), gen = entry_gen(w0), order = entry_order(w0);
                if (h > 1 && q[0].on) {
                    append(q[0], entry_word0(g, h - 1, gen + 1, order), m);
                }
                if (move_x == (order == 0)) {  // on its first leg: turn here into the other axis
                    const uint32_t sp = rt.turn(!move_x, my_r, my_c, m, hbit(g, kTieBit));
                    if (span_fwd(sp) && q[1].on) {
                        append(q[1], entry_word0(g, span_fwd(sp), gen + 1, order), m);
                    }
                    if (span_bwd(sp) && q[2].on) {
                        append(q[2], entry_word0(g, span_bwd(sp), gen + 1, order), m);
                    }
                }
            }
        }
        for (uint32_t k = 0; k < 3; ++k) {
            if (q[k].on) {
                flush_partial(q[k]);
            }
        }
        noc_async_write_barrier();  // rows and entries are in DRAM
        // 3. tails, then 4. the batch's credits
        for (uint32_t k = 0; k < 3; ++k) {
            if (q[k].on && q[k].written != q[k].published) {
                noc_semaphore_inc(
                    get_noc_addr(q[k].tail_noc_x, q[k].tail_noc_y, q[k].tail_sem), q[k].written - q[k].published);
                q[k].published = q[k].written;
            }
        }
        noc_async_atomic_barrier();
        processed += avail;
        noc_semaphore_inc(get_noc_addr(credit_out_sem), avail);
        noc_async_atomic_barrier();
        cy_work += now() - t0;
    }
#ifdef KVD_STATS
    DPRINT(
        "KVDU me={} d={} total={} wait={} work={} batches={} slots={}\n",
        me,
        d,
        now() - t_start,
        cy_wait,
        cy_work,
        n_batches,
        processed);
#endif
    // `arrived` is persistent: leave it at zero for the next call.
    noc_semaphore_inc(get_noc_addr(arrived_addr), 0u - processed);
    noc_semaphore_inc(get_noc_addr(unpack_done_sem), 1);
    noc_async_atomic_barrier();
}
