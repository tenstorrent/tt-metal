// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Accumulator reader: its local rows (the L1 hand-off ring) and its N - 1 rings (one per source chip, each up to this
// call's count in the device plan); a consumed slot is one credit, counted on the port that faces the row's source on
// this accumulator's link to it (that port sends the credits back). Each row goes to the compute with its gate weight.
// It also zeroes this core's state for the call (combine_zero_gate.hpp); accumulator 0's reader is its chip's gate.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/device/kernels/dram_pages.hpp"
#include "combine_zero_gate.hpp"

constexpr uint32_t cb_row = get_compile_time_arg_val(0);
constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
constexpr uint32_t cb_tab = get_compile_time_arg_val(2);
constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
constexpr uint32_t SLOT = get_compile_time_arg_val(4);
constexpr uint32_t PSL = get_compile_time_arg_val(5);  // landing pool slots (one ring per source, carved per call)
constexpr uint32_t A = get_compile_time_arg_val(6);
constexpr uint32_t AP = (A + 3) / 4 * 4;
constexpr uint32_t NT = get_compile_time_arg_val(7);
constexpr uint32_t N = get_compile_time_arg_val(8);
constexpr uint32_t L = get_compile_time_arg_val(9);
constexpr uint32_t RET = get_compile_time_arg_val(10), MISC = get_compile_time_arg_val(11),
                   RINGS = get_compile_time_arg_val(12), LAYO = get_compile_time_arg_val(13);
constexpr uint32_t cb_w = get_compile_time_arg_val(14);
constexpr uint32_t MTAKEN = get_compile_time_arg_val(15), MRTOT = get_compile_time_arg_val(16);
constexpr uint32_t ALAND = get_compile_time_arg_val(17);    // (128, or 160 when ports share the core)
constexpr uint32_t ALBASE = get_compile_time_arg_val(18);   // (101, or 103 when ports share the core)
constexpr uint32_t LOC_OFF = get_compile_time_arg_val(19);  // the local ring (arena-relative)
constexpr uint32_t WORDS_OFF = get_compile_time_arg_val(20);
constexpr uint32_t BAR_SLOT = get_compile_time_arg_val(21);
constexpr uint32_t LASTPB = row_bytes;  // the row's 32 B trailer (one fabric packet a row)

void kernel_main() {
    size_t a = 0;
    const uint32_t area = get_arg_val<uint32_t>(a++);  // the arena (same address on every core and chip)
    const uint32_t plan = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t hc = get_arg_val<uint32_t>(a++);  // this chip's index (its own ring is not there)
    const uint32_t bar = get_arg_val<uint32_t>(a++);
    const uint32_t gate_x = get_arg_val<uint32_t>(a++), gate_y = get_arg_val<uint32_t>(a++);
    // this call's zeroing: the state block and every landing slot's trailer (a slot is taken when its trailer flags
    // are set)
    const uint32_t call = rec::next_call(bar, BAR_SLOT);
    rec::zero_l1(area, RINGS);
    rec::zero_l1_strided(area + RINGS + LASTPB, 32, SLOT, PSL);
    rec::count_in(gate_x, gate_y, bar);
    uint32_t want[N], px[N], py[N];
    for (uint32_t k = 0; k < N; ++k) {
        px[k] = get_arg_val<uint32_t>(a++);
        py[k] = get_arg_val<uint32_t>(a++);
    }
    {
        const uint32_t is_gate = get_arg_val<uint32_t>(a++), nz = get_arg_val<uint32_t>(a++);
        const uint32_t n_cores = get_arg_val<uint32_t>(a++);
        if (is_gate) {
            rec::release(bar, call, nz, n_cores, a);
        }
        a += 2 * n_cores;
    }
    const uint32_t QL = get_arg_val<uint32_t>(a++), WL = get_arg_val<uint32_t>(a++);
    const uint32_t ldid = get_arg_val<uint32_t>(a++);
    const uint32_t LB = area + LOC_OFF, lwords = area + WORDS_OFF;
    const uint32_t wa = a;  // (the y writers' x, y)
    cb_reserve_back(cb_tab, 1);
    uint32_t n_local = 0;
    {  // (the device plan: n_local and want[] of this accumulator)
        const uint32_t sc = (get_write_ptr(cb_tab) + 63) & ~63u;
        const uint64_t plan_noc = dram_pages(plan, 4096).get_noc_addr(0);
        noc_async_read(plan_noc + 64, sc, 64);                        // n_local[16]
        noc_async_read(plan_noc + (32 + me * 32) * 4, sc + 64, 128);  // want[me][32]
        noc_async_read_barrier();
        n_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[me];
        for (uint32_t k = 0; k < N; ++k) {
            want[k] = k == hc ? 0 : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc + 64)[k];
        }
    }
    uint32_t n_rows = n_local;
    for (uint32_t k = 0; k < N; ++k) {
        n_rows += want[k];
    }
    rec::wait_released(bar, call);  // (this chip's ports and writers zeroed: layouts, credits and grants may go out)
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    volatile tt_l1_ptr uint32_t* taken = misc + MTAKEN;  // per ring (source chip), cumulative
    // the landing pool, carved for this call: every source with rows here gets one slot, the rest in proportion to its
    // rows (at most its rows); each source's ring (tag << 24 | base << 12 | depth) goes to the home port that returns
    // its credits, which forwards it to the source (the source's rows wait for it)
    const uint32_t calls_ = misc[105] + 1;
    misc[105] = calls_;
    const uint32_t tag = (calls_ - 1) % 255 + 1;
    uint32_t rbase[N], rdep[N], t0[N];
    {
        uint32_t nz = 0, tw = 0;
        for (uint32_t k = 0; k < N; ++k) {
            if (k != hc && want[k]) {
                ++nz;
                tw += want[k];
            }
        }
        // half the extra slots in equal shares, half in proportion to the rows (all proportional deepened the hot
        // rings past their links' rate and starved the rest: Zipf-spread combine +14%); largest remainders hand out
        // every slot (floors alone left near-equal shares at one slot each)
        const uint32_t eq = nz ? (PSL - nz) / 2 / nz : 0, extra = PSL - nz - eq * nz;
        uint32_t rem[N], left = extra;
        for (uint32_t k = 0; k < N; ++k) {
            rdep[k] = 0;
            rem[k] = 0;
            if (k != hc && want[k]) {
                const uint64_t q = static_cast<uint64_t>(extra) * want[k];
                rdep[k] = 1 + eq + static_cast<uint32_t>(q / tw);
                rem[k] = static_cast<uint32_t>(q % tw);
                left -= rdep[k] - 1 - eq;
            }
        }
        for (; left; --left) {
            uint32_t bk = N;
            for (uint32_t k = 0; k < N; ++k) {
                if (rdep[k] && (bk == N || rem[k] > rem[bk])) {
                    bk = k;
                }
            }
            if (bk == N) {
                break;
            }
            ++rdep[bk];
            rem[bk] = 0;
        }
        uint32_t b = 0;
        for (uint32_t k = 0; k < N; ++k) {
            const uint32_t d = rdep[k] < want[k] ? rdep[k] : want[k];  // (no more slots than rows)
            rbase[k] = b;
            rdep[k] = d;
            b += d;
            t0[k] = taken[k];
            if (k != hc && px[k]) {
                noc_inline_dw_write(
                    get_noc_addr(px[k], py[k], area + LAYO + 4 * (k * AP + me)), tag << 24 | rbase[k] << 12 | d);
            }
        }
    }
    auto ring_slot = [&](uint32_t k, uint32_t tk) { return area + RINGS + (rbase[k] + (tk - t0[k]) % rdep[k]) * SLOT; };
    auto push_row = [&](uint32_t slot, uint32_t t) {  // a row: its gate weight (element 0), the row into the CB
        cb_reserve_back(cb_w, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w))[0] =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + LASTPB)[1] & 0xFFFF;
        cb_push_back(cb_w, 1);
        cb_reserve_back(cb_row, NT);
        cb_reserve_back(cb_meta, 1);
        noc_async_read(get_noc_addr(slot), get_write_ptr(cb_row), row_bytes);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = t / A;
        noc_async_read_barrier();
        cb_push_back(cb_row, NT);
        cb_push_back(cb_meta, 1);
    };
    uint32_t done = 0, li = 0;
    // L1 hand-off: local rows land in a ring of QL slots at LB (the y writers, y_row_major.hpp SE_Y_L1), granted from
    // here
    volatile tt_l1_ptr uint32_t* landed = misc + ALAND;
    const uint32_t lbase = misc[ALBASE];
    uint32_t lgranted = 0;
    auto lgrant_to = [&](uint32_t g) {
        for (uint32_t w = 0; w < WL; ++w) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(wa + 2 * w), get_arg_val<uint32_t>(wa + 1 + 2 * w), lwords + 4 * ldid),
                g - lgranted);
        }
        lgranted = g;
    };
    if (n_local) {
        lgrant_to(n_local < QL ? n_local : QL);
    }
    auto take_local = [&]() -> bool {  // the next local row (L1 hand-off ring), if every writer landed it
        uint32_t m = 0xFFFFFFFF;
        for (uint32_t w = 0; w < WL; ++w) {
            const uint32_t v = landed[w] - lbase;
            m = v < m ? v : m;
        }
        if (li >= m) {
            return false;
        }
        const uint32_t sl = LB + (li % QL) * SLOT;
        push_row(sl, (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl + LASTPB)[0] >> 8) & 0xFFFF);
        ++li;
        ++done;
        const uint32_t g = li + QL < n_local ? li + QL : n_local;
        // (grants in batches of up to 4 - at most the ring: a batch larger than QL never fills, and the writers stall)
        if (g > lgranted && (g - lgranted >= (QL < 4 ? QL : 4) || g == n_local)) {
            lgrant_to(g);
        }
        return true;
    };
    auto take_ring = [&](uint32_t k) -> bool {  // the next row of source k's ring, if it landed
        const uint32_t tk = taken[k];
        const uint32_t s = ring_slot(k, tk);
        volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + LASTPB);
        if (tr[3] == 0 || tr[7] == 0) {
            return false;
        }
        push_row(s, (tr[0] >> 8) & 0xFFFF);
        tr[3] = 0;
        tr[7] = 0;
        taken[k] = tk + 1;
        --want[k];
        ++done;
        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + RET + 4 * (k * AP + me)), 1);
        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + MISC + 4 * (MRTOT + k)), 1);
        return true;
    };
    while (done < n_rows) {
        invalidate_l1_cache();
        while (li < n_local) {
            if (!take_local()) {
                break;
            }
        }
        for (uint32_t k = 0; k < N; ++k) {
            if (want[k]) {
                take_ring(k);
            }
        }
    }
    cb_reserve_back(cb_meta, 1);  // (the compute's end marker: it does not know the row count)
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = 0xFFFFFFFF;
    cb_push_back(cb_meta, 1);
    misc[ALBASE] = lbase + n_local;
    noc_async_atomic_barrier();
}
