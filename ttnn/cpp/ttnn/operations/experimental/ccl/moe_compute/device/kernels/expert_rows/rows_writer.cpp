// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows, second data-movement RISC of one core of one expert group (NC cores). Per job (one expert, at most 32 M
// of its routed rows in M row tiles, from the routing table; e = the job's index in this group's list):
// - x: the job's x moves in chunks of KC hidden tiles (one chunk when KC covers them all). Every group core owns an
//   equal piece of each chunk (so all of them send at once): it reads that slice of each of the job's token rows from
//   the row-major input, rearranges it into tile faces ([hidden tile][row tile], row p of the job = row p % 32 of row
//   tile p / 32) and writes it into slot (chunk % XS) of every group core's cb_x, then increments that slot's
//   semaphore on every group core.
// - a exchange: this core's a tiles (its real intermediate columns of every row tile) to slot (e % NBUF) of every
//   group core's cb_a2, one semaphore per slot.
// - rows: per W2 output group and row tile, the compute leaves 4 bf16 tiles in cb_rows; the job's real rows are copied
//   out of the tile faces and written as posted writes to output row (first row of the job + p), columns of the
//   group's real output tiles.
// Slot reuse. With one x chunk per job and NBUF >= 3 (no credits) the a exchange orders it: once a core holds every
// group core's a(e), every core finished P1(e) (x slot of job e free) and P1(e - 1), hence P2(e - 3). With several
// chunks per job a slot is reused only after every group core handed it back (a credit after its compute popped the
// slot's previous chunk), and the a2 slot the same way; a slot's first use waits for nothing.
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
    constexpr uint32_t ROW_BYTES = get_named_compile_time_arg_val("row_bytes");  // 2 H
    constexpr uint32_t G = get_named_compile_time_arg_val("expert_groups");
    constexpr uint32_t SEM_A2 = get_named_compile_time_arg_val("sem_a2");        // NBUF semaphores
    constexpr uint32_t SEM_X = get_named_compile_time_arg_val("sem_x");          // XS semaphores
    constexpr uint32_t SEM_A2F = get_named_compile_time_arg_val("sem_a2_free");  // with credits: NBUF credits
    constexpr uint32_t SEM_XF = get_named_compile_time_arg_val("sem_x_free");    // with credits: XS credits
    constexpr uint32_t M = get_named_compile_time_arg_val("row_tiles");
    constexpr uint32_t KC = get_named_compile_time_arg_val("chunk_tiles");
    constexpr uint32_t PMAX = get_named_compile_time_arg_val("chunk_piece_max");
    constexpr bool CREDITS = get_named_compile_time_arg_val("slot_credits") == 1;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    static_assert(XS == 1 || XS == 2, "one or two x slots");
    static_assert(CREDITS || NBUF >= 3, "a2 slots without credits");
    static_assert(CREDITS || KC == KT, "x slots without credits: one chunk per job");
    constexpr uint32_t TA = 2048;      // bf16 tile bytes
    constexpr uint32_t RSTR = 4 * 64;  // staging bytes of one row of one output group (4 tiles)
    constexpr uint32_t JR = 32;

    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t na = get_arg_val<uint32_t>(a++);    // real intermediate columns of this core
    const uint32_t c0 = get_arg_val<uint32_t>(a++);    // first of them
    const uint32_t nq = get_arg_val<uint32_t>(a++);    // W2 output groups
    const uint32_t n0 = get_arg_val<uint32_t>(a++);    // first output tile
    const uint32_t nout = get_arg_val<uint32_t>(a++);  // real output tiles
    const uint32_t g = get_arg_val<uint32_t>(a++);
    const uint32_t member = get_arg_val<uint32_t>(a++);  // index of this core in its group
    const uint32_t grp_at = a;                           // mask of this core's expert group
    const uint32_t vx_at = 0, vy_at = GW;                // common args: virtual x / y of the grid

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

    // cb_z: [x staging: JR M rows x PMAX x 64 B][piece tiles: PMAX x M][row staging: JR rows x RSTR]
    constexpr uint32_t PSTR = PMAX * 64;
    const uint32_t stg = get_write_ptr(cb_z);
    const uint32_t tbuf = stg + JR * M * PSTR;
    const uint32_t rstg = tbuf + PMAX * M * TA;
    const auto xs = TensorAccessor(x_args, x_addr, ROW_BYTES);
    const auto out = TensorAccessor(out_args, out_addr, ROW_BYTES);
    const uint32_t x_base = get_write_ptr(cb_x);
    const uint32_t a2_base = get_write_ptr(cb_a2);

    cb_wait_front(cb_rt, 1);
    invalidate_l1_cache();  // the table arrives by NoC read from the root core (rows_reader.cpp)
    const uint32_t rt = get_read_ptr(cb_rt);
    volatile tt_l1_ptr uint16_t* rows = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rt + RWO);
    volatile tt_l1_ptr uint16_t* jobs = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rt + JBO);
    // this group's jobs (local index e): g, g + G, ... with one row tile per job, else group by group as the root
    // dealt them (from the group's first job in the ctl block, rows_reader.cpp)
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt + CTO);
    volatile tt_l1_ptr uint16_t* gs = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rt + CTO + 16);
    const uint32_t nj = ctl[0];
    const uint32_t j0 = M > 1 ? gs[g] : g;
    const uint32_t D = M > 1 ? gs[g + 1] - j0 : (nj > g ? (nj - g + G - 1) / G : 0);
    if (D == 0) {
        return;
    }
    auto J = [&](uint32_t e) -> uint32_t { return M > 1 ? j0 + e : g + G * e; };

    auto write_rows = [&](uint32_t e) {
        const uint32_t j = J(e);
        const uint32_t first = jobs[3 * j + 1], n = jobs[3 * j + 2];
        const uint32_t m = (n + JR - 1) / JR;
        for (uint32_t q = 0; q < nq; ++q) {
            for (uint32_t r = 0; r < m; ++r) {
                cb_wait_front(cb_rows, 4);
                const uint32_t src = get_read_ptr(cb_rows);
                const uint32_t nt = nout - 4 * q < 4 ? nout - 4 * q : 4;
                const uint32_t nr = n - JR * r < JR ? n - JR * r : JR;  // rows of row tile r
                // row p of tile c: 32 B in face ((p / 16) * 2) (columns 0..15) and the next face (16..31), row p % 16
                noc_async_read_one_packet_set_state(get_noc_addr(src), 32);
                for (uint32_t c = 0; c < nt; ++c) {
                    for (uint32_t p = 0; p < nr; ++p) {
                        const uint32_t s0 = src + c * TA + ((p >> 4) * 2) * 512 + (p & 15) * 32;
                        const uint32_t d0 = rstg + p * RSTR + c * 64;
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s0 + 512, d0 + 32);
                    }
                }
                noc_async_read_barrier();
                cb_pop_front(cb_rows, 4);
                // posted writes: non-posted row writes, many in flight under the weight stream, were never acked and
                // hung the device
                const uint32_t col = (n0 + 4 * q) * 64;
                for (uint32_t p = 0; p < nr; ++p) {
                    noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                        rstg + p * RSTR, out.get_noc_addr(first + JR * r + p, col), nt * 64);
                }
                noc_async_posted_writes_flushed();
            }
        }
    };
    // jobs of several row tiles take x in chunks of KC hidden tiles, jobs of one row tile in chunks as large as an x
    // slot of M row tiles holds (rows_compute.cpp)
    constexpr uint32_t KC1 = KC * M < KT ? KC * M : KT;
    auto chunk_tiles_of = [&](uint32_t e) { return jobs[3 * J(e) + 2] > JR ? KC : KC1; };
    auto chunks_of = [&](uint32_t e) {
        const uint32_t kc = chunk_tiles_of(e);
        return (KT + kc - 1) / kc;
    };
    auto wait_sem = [&](uint32_t sem, uint32_t count) {
        volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem));
        while (*v < count) {
            invalidate_l1_cache();
        }
    };
    auto inc_grp = [&](uint32_t sem) {
        const uint32_t l1 = get_semaphore(sem);
        for_grp([&](uint32_t cx, uint32_t cy) { noc_semaphore_inc(get_noc_addr(cx, cy, l1), 1); });
    };
    // this core's piece of chunk c of job e: hidden tiles [c kc + lo, c kc + lo + n)
    auto piece_of = [&](uint32_t e, uint32_t c, uint32_t& lo, uint32_t& n) {
        const uint32_t kc = chunk_tiles_of(e);
        const uint32_t tiles = c + 1 < chunks_of(e) ? kc : KT - c * kc;
        lo = member * tiles / NC;
        n = (member + 1) * tiles / NC - lo;
    };
    // the piece of chunk c for every row of job e, as tiles [piece tile][row tile] in tbuf
    auto build_piece = [&](uint32_t e, uint32_t c) {
        const uint32_t j = J(e);
        const uint32_t first = jobs[3 * j + 1], n = jobs[3 * j + 2];
        const uint32_t m = (n + JR - 1) / JR;
        uint32_t lo, pn;
        piece_of(e, c, lo, pn);
        if (pn == 0) {
            return;
        }
        // staging row stride = this piece (a one-row-tile job's piece is up to M times a held job's)
        const uint32_t k0 = c * chunk_tiles_of(e) + lo, pstr = pn * 64;
        for (uint32_t p = 0; p < n; ++p) {
            noc_async_read(xs.get_noc_addr(rows[first + p], k0 * 64), stg + p * pstr, pn * 64);
        }
        noc_async_read_barrier();
        noc_async_read_one_packet_set_state(get_noc_addr(stg), 32);
        for (uint32_t kk = 0; kk < pn; ++kk) {
            for (uint32_t p = 0; p < n; ++p) {
                const uint32_t s0 = stg + p * pstr + kk * 64;
                const uint32_t d0 = tbuf + (kk * m + p / JR) * TA + (((p % JR) >> 4) * 2) * 512 + (p & 15) * 32;
                noc_async_read_one_packet_with_state(s0, d0);
                noc_async_read_one_packet_with_state(s0 + 32, d0 + 512);
            }
        }
        noc_async_read_barrier();
    };
    // seq: the chunk's number in this core's x stream (slot seq % XS, its use seq / XS)
    auto send_piece = [&](uint32_t e, uint32_t c, uint32_t seq) {
        const uint32_t n = jobs[3 * J(e) + 2];
        const uint32_t m = (n + JR - 1) / JR;
        const uint32_t slot = seq % XS, use = seq / XS;
        if constexpr (CREDITS) {
            cb_reserve_back(cb_x, KC * M);  // this core's compute popped the slot's previous chunk
            if (use > 0) {  // a slot starts free on every group core: only reuses wait for the hand-back
                inc_grp(SEM_XF + slot);
                wait_sem(SEM_XF + slot, NC * use);
            }
        }
        uint32_t lo, pn;
        piece_of(e, c, lo, pn);
        if (pn) {
            const uint32_t dst = x_base + (slot * KC * M + lo * m) * TA;
            if (m == 1 && n <= 16) {  // rows 0..15 only: the top two faces of each tile
                for_grp([&](uint32_t cx, uint32_t cy) {
                    for (uint32_t kk = 0; kk < pn; ++kk) {
                        noc_async_write(tbuf + kk * TA, get_noc_addr(cx, cy, dst + kk * TA), n * 32);
                        noc_async_write(tbuf + kk * TA + 512, get_noc_addr(cx, cy, dst + kk * TA + 512), n * 32);
                    }
                });
            } else {
                for_grp(
                    [&](uint32_t cx, uint32_t cy) { noc_async_write(tbuf, get_noc_addr(cx, cy, dst), pn * m * TA); });
            }
            noc_async_write_barrier();
        }
        inc_grp(SEM_X + slot);
    };
    auto push_chunk = [&](uint32_t seq) {
        const uint32_t slot = seq % XS, use = seq / XS;
        wait_sem(SEM_X + slot, NC * (use + 1));
        if constexpr (!CREDITS) {
            cb_reserve_back(cb_x, KC * M);
        }
        cb_push_back(cb_x, KC * M);
    };
    auto exchange = [&](uint32_t e) {
        const uint32_t n = jobs[3 * J(e) + 2];
        const uint32_t m = (n + JR - 1) / JR;
        const uint32_t slot = e % NBUF, use = e / NBUF;
        cb_wait_front(cb_a, M * a_tiles);
        const uint32_t src = get_read_ptr(cb_a);
        if constexpr (CREDITS) {
            cb_reserve_back(cb_a2, M * Nt);  // this core's compute popped the slot's previous job
            if (use > 0) {
                inc_grp(SEM_A2F + slot);
                wait_sem(SEM_A2F + slot, NC * use);
            }
        }
        for_grp([&](uint32_t cx, uint32_t cy) {
            for (uint32_t r = 0; r < m; ++r) {
                noc_async_write(
                    src + r * a_tiles * TA,
                    get_noc_addr(cx, cy, a2_base + (slot * M * Nt + r * Nt + c0) * TA),
                    na * TA);
            }
        });
        noc_async_write_barrier();
        inc_grp(SEM_A2 + slot);
        cb_pop_front(cb_a, M * a_tiles);
        wait_sem(SEM_A2 + slot, NC * (use + 1));
        if constexpr (!CREDITS) {
            cb_reserve_back(cb_a2, M * Nt);
        }
        cb_push_back(cb_a2, M * Nt);
    };

    if constexpr (!CREDITS) {
        // One chunk per job: job e's x goes to slot e % XS after exchange(e - XS) (every core finished P1(e - XS));
        // job e - 1's rows are drained after exchange(e): the compute packs them in P2(e - 1), right after P1(e).
        for (uint32_t e = 0; e < XS && e < D; ++e) {
            build_piece(e, 0);
            send_piece(e, 0, e);
        }
        for (uint32_t e = 0; e < XS && e < D; ++e) {
            push_chunk(e);
        }
        for (uint32_t e = 0; e < D; ++e) {
            exchange(e);
            if (e + XS < D) {
                build_piece(e + XS, 0);
                send_piece(e + XS, 0, e + XS);
                push_chunk(e + XS);
            }
            if (e > 0) {
                write_rows(e - 1);
            }
        }
        write_rows(D - 1);
        noc_async_full_barrier();
        return;
    }

    // Several chunks per job: one stream of x chunks over all of this group's jobs; each chunk's piece is built while
    // the other group cores send theirs. Compute runs P1(0), P1(1), P2(0), P1(2), P2(1), ...: once the first XS chunks
    // of job j + 1 are in (their slots were freed by P1(j)), drain the rows of P2(j - 1) and exchange a(j), which with
    // one a2 slot needs P2(j - 1) done on every group core. The next job's first chunks are then already in place when
    // P2 ends.
    uint32_t seq = 0;
    build_piece(0, 0);
    for (uint32_t e = 0; e < D; ++e) {
        const uint32_t ce = chunks_of(e);
        const uint32_t trigger = (XS < ce ? XS : ce) - 1;
        for (uint32_t c = 0; c < ce; ++c) {
            send_piece(e, c, seq);
            if (c + 1 < ce) {
                build_piece(e, c + 1);
            } else if (e + 1 < D) {
                build_piece(e + 1, 0);
            }
            push_chunk(seq++);
            if (c == trigger && e >= 1) {
                if (e >= 2) {
                    write_rows(e - 2);
                }
                exchange(e - 1);
            }
        }
    }
    if (D >= 2) {
        write_rows(D - 2);
    }
    exchange(D - 1);
    write_rows(D - 1);
    noc_async_full_barrier();
}
