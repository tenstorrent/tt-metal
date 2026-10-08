// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows, second data-movement RISC of one core of one expert group (NC cores). Per job j of the group
// (one expert, at most 32 of its routed rows, from the routing table):
// - x gather: this core owns hidden tiles [x0, x0 + xn); it reads that slice of each of the job's token rows from the
//   row-major input, rearranges it into tile faces (row p of the job = row p of the tile) and writes the real rows
//   into slot (j % XS) of every group core's cb_x, then increments that slot's semaphore on every group core.
// - a exchange: this core's a tiles (its real intermediate columns) to slot (j % NBUF) of every group core's cb_a2,
//   one semaphore per slot. NBUF >= 3: when a core sends a(j) it only knows that every core finished P1(j - 1),
//   hence P2(j - 3).
// - rows: per W2 output group, the compute leaves 4 bf16 tiles in cb_rows; the job's real rows are copied out of the
//   tile faces and written as posted writes to output row (first row of the job + p), columns of the group's real
//   output tiles.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_named_compile_time_arg_val("cb_a");
    constexpr uint32_t cb_a2 = get_named_compile_time_arg_val("cb_a2");
    constexpr uint32_t cb_rt = get_named_compile_time_arg_val("cb_rt");
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_z = get_named_compile_time_arg_val("cb_z");
    constexpr uint32_t cb_rows = get_named_compile_time_arg_val("cb_rows");
    constexpr uint32_t NC = get_named_compile_time_arg_val("group_cores");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("intermediate_tiles");
    constexpr uint32_t KT = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t NBUF = get_named_compile_time_arg_val("a2_slots");
    constexpr uint32_t XS = get_named_compile_time_arg_val("x_slots");
    constexpr uint32_t a_tiles = get_named_compile_time_arg_val("a_tiles");
    constexpr uint32_t CTO = get_named_compile_time_arg_val("rt_ctl");
    constexpr uint32_t RWO = get_named_compile_time_arg_val("rt_rows");
    constexpr uint32_t JBO = get_named_compile_time_arg_val("rt_jobs");
    constexpr uint32_t GW = get_named_compile_time_arg_val("grid_w");
    constexpr uint32_t GH = get_named_compile_time_arg_val("grid_h");
    constexpr uint32_t MW = get_named_compile_time_arg_val("mask_words");
    constexpr uint32_t XNMAX = get_named_compile_time_arg_val("x_tiles_max");
    constexpr uint32_t ROW_BYTES = get_named_compile_time_arg_val("row_bytes");  // 2 H
    constexpr uint32_t G = get_named_compile_time_arg_val("expert_groups");
    constexpr uint32_t SEM_A2 = get_named_compile_time_arg_val("sem_a2");  // NBUF semaphores
    constexpr uint32_t SEM_X = get_named_compile_time_arg_val("sem_x");    // XS semaphores
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    static_assert(XS == 1 || XS == 2, "one or two x slots");
    static_assert(NBUF >= 3, "a2 slots");
    constexpr uint32_t TA = 2048;          // bf16 tile bytes
    constexpr uint32_t RSTR = 4 * 64;      // staging bytes of one row of one output group (4 tiles)
    constexpr uint32_t XSTR = XNMAX * 64;  // staging bytes of one row of this core's x slice
    constexpr uint32_t JR = 32;

    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t x0 = get_arg_val<uint32_t>(a++);
    const uint32_t xn = get_arg_val<uint32_t>(a++);
    const uint32_t na = get_arg_val<uint32_t>(a++);    // real intermediate columns of this core
    const uint32_t c0 = get_arg_val<uint32_t>(a++);    // first of them
    const uint32_t nq = get_arg_val<uint32_t>(a++);    // W2 output groups
    const uint32_t n0 = get_arg_val<uint32_t>(a++);    // first output tile
    const uint32_t nout = get_arg_val<uint32_t>(a++);  // real output tiles
    const uint32_t g = get_arg_val<uint32_t>(a++);
    const uint32_t grp_at = a;             // mask of this core's expert group
    const uint32_t vx_at = 0, vy_at = GW;  // common args: virtual x / y of the grid

    auto for_grp = [&](auto&& fn) {
        for (uint32_t w = 0; w < MW; ++w) {
            uint32_t bits = get_arg_val<uint32_t>(grp_at + w);
            while (bits) {
                const uint32_t b = __builtin_ctz(bits);
                bits &= bits - 1;
                const uint32_t li = w * 32 + b;
                fn(get_common_arg_val<uint32_t>(vx_at + li / GH), get_common_arg_val<uint32_t>(vy_at + li % GH));
            }
        }
    };

    // cb_z: [x staging: JR rows x XSTR][tile build: XNMAX tiles][row staging: JR rows x RSTR]
    const uint32_t stg = get_write_ptr(cb_z);
    const uint32_t tbuf = stg + JR * XSTR;
    const uint32_t rstg = tbuf + XNMAX * TA;
    const auto xs = TensorAccessor(x_args, x_addr, ROW_BYTES);
    const auto out = TensorAccessor(out_args, out_addr, ROW_BYTES);
    const uint32_t x_base = get_write_ptr(cb_x);
    const uint32_t a2_base = get_write_ptr(cb_a2);

    cb_wait_front(cb_rt, 1);
    invalidate_l1_cache();  // the table arrives by NoC read from the root core (rows_reader.cpp)
    const uint32_t rt = get_read_ptr(cb_rt);
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt + CTO);
    volatile tt_l1_ptr uint16_t* rows = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rt + RWO);
    volatile tt_l1_ptr uint16_t* jobs = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rt + JBO);
    const uint32_t nj = ctl[0];
    const uint32_t D = nj > g ? (nj - g + G - 1) / G : 0;  // this group's jobs g, g + G, ... (local index e)
    if (D == 0) {
        return;
    }
    auto J = [&](uint32_t e) -> uint32_t { return g + G * e; };

    auto gather_x = [&](uint32_t e) {
        const uint32_t j = J(e);
        const uint32_t first = jobs[3 * j + 1], n = jobs[3 * j + 2];
        if (xn) {
            for (uint32_t p = 0; p < n; ++p) {
                noc_async_read(xs.get_noc_addr(rows[first + p], x0 * 64), stg + p * XSTR, xn * 64);
            }
            noc_async_read_barrier();
            // staging row p, hidden tile kk (64 B) -> face rows p of the two column faces of tile kk (local NoC
            // copies; columns 16..31 are 512 B further)
            noc_async_read_one_packet_set_state(get_noc_addr(stg), 32);
            for (uint32_t kk = 0; kk < xn; ++kk) {
                for (uint32_t p = 0; p < n; ++p) {
                    const uint32_t s0 = stg + p * XSTR + kk * 64;
                    const uint32_t d0 = tbuf + kk * TA + ((p >> 4) * 2) * 512 + (p & 15) * 32;
                    noc_async_read_one_packet_with_state(s0, d0);
                    noc_async_read_one_packet_with_state(s0 + 32, d0 + 512);
                }
            }
            noc_async_read_barrier();
        }
        const uint32_t slot = (e % XS) * KT;
        for_grp([&](uint32_t cx, uint32_t cy) {
            for (uint32_t kk = 0; kk < xn; ++kk) {
                const uint32_t src = tbuf + kk * TA;
                const uint32_t dst = x_base + (slot + x0 + kk) * TA;
                if (n <= 16) {
                    noc_async_write(src, get_noc_addr(cx, cy, dst), n * 32);
                    noc_async_write(src + 512, get_noc_addr(cx, cy, dst + 512), n * 32);
                } else {
                    noc_async_write(src, get_noc_addr(cx, cy, dst), TA);
                }
            }
        });
        noc_async_write_barrier();
        const uint32_t sem = get_semaphore(SEM_X + (e % XS));
        for_grp([&](uint32_t cx, uint32_t cy) { noc_semaphore_inc(get_noc_addr(cx, cy, sem), 1); });
    };
    auto push_x = [&](uint32_t e) {
        volatile tt_l1_ptr uint32_t* s =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_X + (e % XS)));
        while (*s < NC * (e / XS + 1)) {
            invalidate_l1_cache();
        }
        cb_reserve_back(cb_x, KT);
        cb_push_back(cb_x, KT);
    };
    auto write_rows = [&](uint32_t e) {
        const uint32_t j = J(e);
        const uint32_t first = jobs[3 * j + 1], n = jobs[3 * j + 2];
        for (uint32_t q = 0; q < nq; ++q) {
            cb_wait_front(cb_rows, 4);
            const uint32_t src = get_read_ptr(cb_rows);
            const uint32_t nt = nout - 4 * q < 4 ? nout - 4 * q : 4;
            // row p of tile c: 32 B in face ((p / 16) * 2) (columns 0..15) and the next face (16..31), row p % 16
            noc_async_read_one_packet_set_state(get_noc_addr(src), 32);
            for (uint32_t c = 0; c < nt; ++c) {
                for (uint32_t p = 0; p < n; ++p) {
                    const uint32_t s0 = src + c * TA + ((p >> 4) * 2) * 512 + (p & 15) * 32;
                    const uint32_t d0 = rstg + p * RSTR + c * 64;
                    noc_async_read_one_packet_with_state(s0, d0);
                    noc_async_read_one_packet_with_state(s0 + 512, d0 + 32);
                }
            }
            noc_async_read_barrier();
            cb_pop_front(cb_rows, 4);
            // posted writes: non-posted row writes, many in flight under the weight stream, were never acked and hung
            // the device
            const uint32_t col = (n0 + 4 * q) * 64;
            for (uint32_t p = 0; p < n; ++p) {
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                    rstg + p * RSTR, out.get_noc_addr(first + p, col), nt * 64);
            }
            noc_async_posted_writes_flushed();
        }
    };
    auto exchange = [&](uint32_t e) {
        cb_wait_front(cb_a, a_tiles);
        const uint32_t src = get_read_ptr(cb_a);
        const uint32_t par = (e % NBUF) * Nt;
        const uint32_t sem_l1 = get_semaphore(SEM_A2 + (e % NBUF));
        for_grp([&](uint32_t cx, uint32_t cy) {
            noc_async_write(src, get_noc_addr(cx, cy, a2_base + (par + c0) * TA), na * TA);
        });
        noc_async_write_barrier();
        for_grp([&](uint32_t cx, uint32_t cy) { noc_semaphore_inc(get_noc_addr(cx, cy, sem_l1), 1); });
        cb_pop_front(cb_a, a_tiles);
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1);
        do {
            invalidate_l1_cache();
        } while (*sem < NC * (e / NBUF + 1));
        cb_reserve_back(cb_a2, Nt);
        cb_push_back(cb_a2, Nt);
    };

    for (uint32_t e = 0; e < XS && e < D; ++e) {
        gather_x(e);
    }
    for (uint32_t e = 0; e < XS && e < D; ++e) {
        push_x(e);
    }
    for (uint32_t e = 0; e < D; ++e) {
        exchange(e);
        if (e + XS < D) {
            gather_x(e + XS);
            push_x(e + XS);
        }
        // job e - 1's rows: compute packs them in P2(e - 1), right after P1(e), whose a tiles exchange(e) took
        if (e > 0) {
            write_rows(e - 1);
        }
    }
    write_rows(D - 1);
    noc_async_full_barrier();
}
