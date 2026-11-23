// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_kv_dispatch v3 TX(d, l) reader of one lane (the BRISC writer, kvd3_tx.cpp, owns the fabric connection). As
// sparse_dispatch_2d splits its port, this kernel does every per-packet decision so the writer only sends:
//   - choose a published staging slot of this lane's producers (planners carry class 0, my workers W(f, l, k) the
//     rest) whose class has a free landing slot on one of this lane's neighbour workers (credits: the neighbour TX's
//     snapshots in this core's control block), NoC-read it into the next free ring buffer, free it on its producer,
//     and hand the buffer to the writer with its destination (worker, landing slot address, class, payload);
//   - close class c (an END(c) buffer for each of this lane's workers) once every producer of the lane is done with c
//     and every slot it published is in the ring; classes close in order;
//   - finish once every class is closed and every packet handed out has been credited back by the neighbour.
// Lanes (lanes3 = 2): producers split by index parity, the neighbour's workers in halves; lane 0 runs here on the TX
// core's NCRISC (ring A in the send CB, program-semaphore counters), lane 1 on a helper core (PUSH): it reads into its
// own buffers and writes each finished buffer and descriptor into ring B on the TX core, then raises kTxFilledB there;
// the writer returns drained counts to its kHDrained word.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "kvd3_common.hpp"

#include <cstdint>

using namespace kvd;
using namespace kvd3;

#ifdef KVD_FUSED_PLAN
// the fused plan (kvd4): this kernel runs the chip's dispatch-only plan first (sparse_push_plan.cpp, PLAN_FUSED)
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/plan/device/kernels/sparse_push_plan.cpp"
#endif

void kernel_main() {
#ifdef KVD_FUSED_PLAN
    plan_prologue();
#endif
    constexpr uint32_t L = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t P = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr uint32_t B = get_compile_time_arg_val(4);
    constexpr uint32_t pool = get_compile_time_arg_val(5);
    constexpr uint32_t stage = get_compile_time_arg_val(6);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(8);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(9);
    constexpr uint32_t cb_pool = get_compile_time_arg_val(10);
    constexpr uint32_t cb_send = get_compile_time_arg_val(11);
    constexpr uint32_t sem_go = get_compile_time_arg_val(12);
    constexpr uint32_t sem_filled = get_compile_time_arg_val(13);
    constexpr uint32_t sem_drained = get_compile_time_arg_val(14);
    constexpr uint32_t sem_rdone = get_compile_time_arg_val(15);
    constexpr uint32_t pstage = get_compile_time_arg_val(16);   // planners' staging slots a port
    constexpr uint32_t lanes3 = get_compile_time_arg_val(17);   // 1 or 2
    constexpr uint32_t lane = get_compile_time_arg_val(18);     // this reader's lane
    constexpr bool push = get_compile_time_arg_val(19) != 0;    // a helper core: push into ring B of the TX core
    constexpr uint32_t n_cores = get_compile_time_arg_val(20);  // (push: the chip barrier's count)
    constexpr uint32_t sbytes = slot_bytes(B, row_bytes);
    // (KVD_SEQ with LANES=2: each lane feeds its own landing workers kb..ke, so a (worker, class) has one sequence; a
    // helper stamps its buffer before pushing it into ring B)
    constexpr uint32_t rows_off = slot_rows_offset(B);
    constexpr uint32_t kb = lane * K / lanes3, ke = (lane + 1) * K / lanes3;  // this lane's neighbour workers
    constexpr uint32_t pstep = lanes3;                                        // this lane's producers: p % lanes3
    static_assert(lanes3 == 1 || (P % 2 == 0 && K % 2 == 0));
    static_assert(kRing <= 15, "streaming reads use transaction id 1 + buffer (ids 1..15)");
#ifdef KVD_ABL_HUNGRY
    static_assert(KVD_ABL_HUNGRY >= 1, "HUNGRY=n: n empty picks between flush requests");
#ifdef KVD_ABL_BATCH_READS
#error "HUNGRY is implemented on the streaming reader only (not with BATCH_READS)"
#endif
#endif
    (void)n_cores;

    uint32_t a = 0;
    const uint32_t present = get_arg_val<uint32_t>(a++);
    const uint32_t ctrl = get_write_ptr(cb_ctrl);
    auto* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl);
    uint32_t tx_x = 0, tx_y = 0;
    if constexpr (push) {
        // a core of its own: zero the control block and take part in the chip barrier (as the workers)
        const uint32_t bar_x = get_arg_val<uint32_t>(a++);
        const uint32_t bar_y = get_arg_val<uint32_t>(a++);
#ifdef KVD_ABL_ZERO_LOOP
        for (uint32_t i = 0; i < kCtrlWords; ++i) {
            cw[i] = 0;
        }
#else
        zero_ctrl(ctrl);
#endif
        noc_semaphore_inc(get_noc_addr(bar_x, bar_y, get_common_arg_val<uint32_t>(10)), 1);
        noc_async_atomic_barrier();
        auto* bgo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_val<uint32_t>(9));
        noc_semaphore_wait_min(bgo, 1);
        noc_semaphore_set(bgo, 0);
        tx_x = get_arg_val<uint32_t>(a++);
        tx_y = get_arg_val<uint32_t>(a++);
    }
    if (!present) {
        return;
    }
    const uint32_t d = get_arg_val<uint32_t>(a++);
    const uint32_t l = get_arg_val<uint32_t>(a++);
    uint32_t prod_x[P + 4 * K], prod_y[P + 4 * K];
    for (uint32_t p = 0; p < P + 4 * K; ++p) {
        prod_x[p] = get_arg_val<uint32_t>(a++);
        prod_y[p] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t n_workers = get_arg_val<uint32_t>(a++);

    if constexpr (!push) {
        // the writer releases this kernel once the chip barrier is passed (the control block is zeroed)
        auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go));
        while (*go == 0) {
        }
#ifdef KVD_ACTIVE_FROM_BLOCK
        if (*go == 2) {  // the writer found this chunk before active_from
            return;
        }
#endif
    }
    // ring counters: lane 0 program semaphores on this (TX) core; push: kHDrained here, kTxFilledB on the TX core
    auto* drained_w =
        push ? cw + kHDrained : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_drained));
    auto* filled_w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_filled));
    const uint32_t stage_base = align64(get_write_ptr(cb_stage));
    const uint32_t pool_base = align64(get_write_ptr(cb_pool));
    const uint32_t send_l1 = align64(get_write_ptr(cb_send));  // (the same address on the TX core)
    // this lane's buffers and descriptors in the TX core's layout: ring A at 0, ring B at kRing
    constexpr uint32_t ring0 = push ? kRing : 0;
    auto buf_l1 = [&](uint32_t i) { return send_l1 + (ring0 + i) * sbytes; };
    const uint32_t desc0 = desc_l1(send_l1, sbytes) + ring0 * 16;
    auto* desc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(desc0);

    uint32_t sent[K][G], pulled[P + 4 * K][G], cls_pulled[G], rr[G], cursor[G];
    for (uint32_t c = 0; c < G; ++c) {
        cls_pulled[c] = 0;
        rr[c] = kb;
        cursor[c] = 0;
        for (uint32_t k = 0; k < K; ++k) {
            sent[k][c] = 0;
        }
        for (uint32_t p = 0; p < P + 4 * K; ++p) {
            pulled[p][c] = 0;
        }
    }
    auto pick_worker = [&](uint32_t c) -> int32_t {
        uint32_t k = rr[c];
        for (uint32_t n = kb; n < ke; ++n) {
            if (sent[k][c] - cw[tx_credit(k, c)] < pool_depth(c, pool)) {
                rr[c] = k + 1 == ke ? kb : k + 1;
                return static_cast<int32_t>(k);
            }
            k = k + 1 == ke ? kb : k + 1;
        }
        return -1;
    };
    uint32_t filled = 0;
    uint32_t pend_seq[kRing];
    // a buffer's destination is taken when it is chosen (sent[k][c] reserves the landing slot), handed over later
    auto reserve = [&](uint32_t i, uint32_t k, uint32_t c) {
        desc[i * 4 + 0] = k;
        desc[i * 4 + 1] = c;
        desc[i * 4 + 2] = pool_base + pool_offset(c, sent[k][c] % pool_depth(c, pool), pool, sbytes);
        ++sent[k][c];
        pend_seq[i] = kSeqMagic | sent[k][c];
    };
    // KVD_SEQ: stamp a buffer whose contents are final (header word 3 and its last 16 B)
    auto stamp = [&](uint32_t i, uint32_t n, bool end) {
#ifdef KVD_SEQ
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_l1(i))[kHdrSeq] = pend_seq[i];
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_l1(i) + seq_trailer(B, row_bytes, n, end))[0] = pend_seq[i];
#endif
    };
    (void)stamp;
    uint32_t pend_p[kRing], pend_c[kRing];
    uint32_t cy_pick = 0, cy_rd = 0, cy_tail = 0;
    uint32_t total_pulled = 0;
    // class c: one of this lane's producers with a published slot (from the cursor on), if c has a landing slot
    uint32_t n_try = 0, n_scan = 0;  // (KVD_STATS: try_class calls, producer slots scanned)
    auto try_class = [&](uint32_t c, uint32_t i) -> bool {
#ifdef KVD_STATS
        ++n_try;
#endif
        if (cw[tx_cls_pub(lane, c)] == cls_pulled[c]) {
            return false;
        }
        const uint32_t lo = (c == 0 ? 0 : P) + lane, hi = c == 0 ? P : P + 4 * K;  // p = lo, lo + pstep, ...
        const uint32_t count = (hi - lo + pstep - 1) / pstep;
        uint32_t j = cursor[c];
        for (uint32_t n = 0; n < count; ++n) {
#ifdef KVD_STATS
            ++n_scan;
#endif
            const uint32_t p = lo + j * pstep;
            if (cw[tx_pub(p, c)] != pulled[p][c]) {
                const uint32_t q0 = stats_clock();
                const int32_t k = pick_worker(c);
                const uint32_t q1 = stats_clock();
                cy_pick += q1 - q0;
                if (k < 0) {
                    return false;  // no landing slot for this class
                }
                const uint32_t off = c == 0 ? stage_offset(d * L + l, 0, pulled[p][c] % pstage, 1, pstage, sbytes)
                                            : stage_offset(d, c, pulled[p][c] % stage, G, stage, sbytes);
#ifdef KVD_ABL_R_HDR_ONLY
                noc_async_read_set_trid(1 + i);
                noc_async_read(get_noc_addr(prod_x[p], prod_y[p], stage_base + off), buf_l1(i), rows_off);
#else
#ifdef KVD_ABL_BATCH_READS
                noc_async_read(get_noc_addr(prod_x[p], prod_y[p], stage_base + off), buf_l1(i), sbytes);
#else
                const uint32_t q2 = stats_clock();
                noc_async_read_set_trid(1 + i);  // (each ring buffer's read is tracked on its own)
                noc_async_read(get_noc_addr(prod_x[p], prod_y[p], stage_base + off), buf_l1(i), sbytes);
                const uint32_t q3 = stats_clock();
                cy_rd += q3 - q2;
#endif
#endif
                const uint32_t q4 = stats_clock();
                ++pulled[p][c];
                ++cls_pulled[c];
                ++total_pulled;
                cursor[c] = cw[tx_pub(p, c)] != pulled[p][c] ? j : (j + 1 == count ? 0 : j + 1);
                reserve(i, static_cast<uint32_t>(k), c);
                pend_p[i] = p;
                pend_c[i] = c;
                cy_tail += stats_clock() - q4;
                return true;
            }
            j = j + 1 == count ? 0 : j + 1;
        }
        return false;
    };
    uint32_t last_c = G - 1;
    auto choose = [&](uint32_t i) -> bool {
        if (cw[tx_pub_total(lane)] == total_pulled) {  // nothing new published in this lane
            return false;
        }
#ifdef KVD_ABL_CLASS0_FIRST
        if (try_class(0, i)) {  // planners first: class 0 closes (and its END wave starts) early
            return true;
        }
#endif
        if (try_class(last_c, i)) {
            return true;
        }
        for (uint32_t ci = 0; ci < G; ++ci) {
            const uint32_t c = G - 1 - ci;  // deepest first
            if (try_class(c, i)) {
                last_c = c;
                return true;
            }
        }
        return false;
    };
    // hand n buffers (from `filled` on) to the writer: push writes them (and their descriptors) into ring B first
    auto hand_over = [&](uint32_t n) {
        if constexpr (push) {
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t i = (filled + j) % kRing;
                noc_async_write(buf_l1(i), get_noc_addr(tx_x, tx_y, buf_l1(i)), desc[i * 4 + 3]);
                noc_async_write(desc0 + i * 16, get_noc_addr(tx_x, tx_y, desc0 + i * 16), 16);
            }
            noc_async_write_barrier();  // buffers and descriptors are on the TX core before it hears of them
            noc_semaphore_inc(get_noc_addr(tx_x, tx_y, ctrl + kTxFilledB * 4), n);
        }
        filled += n;
        if constexpr (!push) {
            *filled_w = filled;
        }
    };
    (void)filled_w;

    uint32_t closed = 0, end_sent = kb;
    uint32_t tl_ready[8] = {0, 0, 0, 0, 0, 0, 0, 0}, tl_closed[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // (KVD_STATS, 0.1 us)
    (void)tl_ready;
    (void)tl_closed;
    uint32_t spins = 0;
    (void)spins;
    const uint32_t t0 = stats_clock();
    uint32_t cy_choose = 0, cy_bar = 0, cy_hand = 0, cy_none = 0, cy_full = 0, n_batches = 0, n_pk = 0;
    uint32_t issued = 0;  // buffers chosen (reads issued): filled..issued-1 are in flight
    uint32_t cy_ok = 0, cy_fail = 0, n_ok = 0, n_fail = 0, n_why_empty = 0, n_why_credit = 0, empty_streak = 0;
    // (KVD_STATS) Little's law: every 16th loop, the published-but-unpulled backlog of this lane and the landing slots
    // in flight (sent, credit not back) over this lane's workers and classes
    uint32_t n_samp = 0, n_samp_backlog = 0, sum_backlog = 0, sum_inflight = 0, n_samp_full = 0, loop_i = 0;
    uint32_t sum_backlog0 = 0, sum_inflight0 = 0, n_samp_c0_capped = 0;  // class 0 alone
    (void)sum_backlog0;
    (void)sum_inflight0;
    (void)n_samp_c0_capped;
    (void)n_samp;
    (void)n_samp_backlog;
    (void)sum_backlog;
    (void)sum_inflight;
    (void)n_samp_full;
    (void)loop_i;
    while (true) {
#ifdef KVD_STATS
        if ((++loop_i & 15) == 0 && closed == 0) {  // (class-0 phase only)
            const uint32_t backlog = cw[tx_pub_total(lane)] - total_pulled;
            uint32_t inflight = 0;
            for (uint32_t k = kb; k < ke; ++k) {
                for (uint32_t c = 0; c < G; ++c) {
                    inflight += sent[k][c] - cw[tx_credit(k, c)];
                }
            }
            ++n_samp;
            n_samp_backlog += backlog != 0;
            sum_backlog += backlog;
            sum_inflight += inflight;
            n_samp_full += (issued - *drained_w) >= kRing;  // (as admission counts: reads in flight too)
            const uint32_t b0 = cw[tx_cls_pub(lane, 0)] - cls_pulled[0];
            uint32_t i0 = 0;
            for (uint32_t k = kb; k < ke; ++k) {
                i0 += sent[k][0] - cw[tx_credit(k, 0)];
            }
            sum_backlog0 += b0;
            sum_inflight0 += i0;
            n_samp_c0_capped += (b0 != 0 && i0 >= (ke - kb) * pool_depth(0, pool));
        }
#endif
#ifdef KVD_ABL_BATCH_READS
        // every free ring buffer: choose and start its read; one barrier; then free the producers and hand over
        const uint32_t room = kRing - (filled - *drained_w);
        uint32_t n = 0;
        const uint32_t tc = stats_clock();
        while (n < room && choose((filled + n) % kRing)) {
            ++n;
        }
        const uint32_t tb = stats_clock();
        if (n) {
            cy_choose += tb - tc;
            ++n_batches;
            n_pk += n;
            noc_async_read_barrier();
            const uint32_t th = stats_clock();
            cy_bar += th - tb;
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t i = (filled + j) % kRing, p = pend_p[i], c = pend_c[i];
                const uint32_t freed_word = c == 0 ? p_freed(d * L + l) : w_freed(d, c);
                noc_semaphore_inc(get_noc_addr(prod_x[p], prod_y[p], ctrl + freed_word * 4), 1);
                const uint32_t entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_l1(i))[kHdrEntries];
                stamp(i, entries, false);
                desc[i * 4 + 3] = rows_off + entries * row_bytes;
            }
            hand_over(n);
            cy_hand += stats_clock() - th;
            continue;
        }
        issued = filled;
#else
        // streaming: keep every free ring buffer's read in flight, and hand buffers over in ring order as soon as
        // their own read (transaction id 1 + buffer) has landed: the writer starts on the first, not the last
        const uint32_t tc = stats_clock();
        bool moved = false;
        while (filled != issued) {
            const uint32_t i = filled % kRing;
            while (!noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
            }
            if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + i)) {
                break;
            }
            const uint32_t p = pend_p[i], c = pend_c[i];
            const uint32_t freed_word = c == 0 ? p_freed(d * L + l) : w_freed(d, c);
            noc_semaphore_inc(get_noc_addr(prod_x[p], prod_y[p], ctrl + freed_word * 4), 1);
            const uint32_t entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_l1(i))[kHdrEntries];
            stamp(i, entries, false);
            desc[i * 4 + 3] = rows_off + entries * row_bytes;
            hand_over(1);
            ++n_pk;
            moved = true;
        }
        const uint32_t th = stats_clock();
        cy_hand += th - tc;
        while (issued - *drained_w < kRing) {
            const uint32_t t1 = stats_clock();
            const bool ok = choose(issued % kRing);
            const uint32_t t2 = stats_clock();
            if (!ok) {
                cy_fail += t2 - t1;
                ++n_fail;
                // why: nothing published in this lane, or published but no landing credit downstream
                if (cw[tx_pub_total(lane)] == total_pulled) {
                    ++n_why_empty;
#ifdef KVD_ABL_HUNGRY
                    // starving: every KVD_ABL_HUNGRY empty picks, ask this port's worker producers for their open
                    // (partial) staging slots
                    if (++empty_streak % KVD_ABL_HUNGRY == 0) {
                        for (uint32_t p = P + lane; p < P + 4 * K; p += pstep) {
                            if (prod_x[p] != 0 || prod_y[p] != 0) {
                                noc_semaphore_inc(get_noc_addr(prod_x[p], prod_y[p], ctrl + w_flush_req(d) * 4), 1);
                            }
                        }
                    }
#endif
                } else {
                    ++n_why_credit;
                }
                break;
            }
            cy_ok += t2 - t1;
            ++n_ok;
            empty_streak = 0;
            ++issued;
            moved = true;
        }
        cy_choose += stats_clock() - th;
#ifdef KVD_ABL_EAGER_END
        if (filled != issued) {  // (a class may close while others still move: only reads in flight block it)
#else
        if (moved || filled != issued) {
#endif
            continue;  // (reads in flight: no class may close behind them)
        }
        const uint32_t room = kRing - (filled - *drained_w);
#endif
        if (room == 0) {
            cy_full += stats_clock() - tc;
            continue;  // the ring is full: the writer is behind
        }
        cy_none += stats_clock() - tc;
#ifdef KVD_ABL_SPIN_DEBUG
        if ((++spins & 0xFFFFF) == 0 && d == 0 && l == 0) {
            DPRINT(
                "SPIN lane={} closed={} filled={} drained={} pubtot={} pulled={} cp0={} cl0={} dn0={} cp1={} cl1={} "
                "dn1={} "
                "cr0={} sent0={}\n",
                lane,
                closed,
                filled,
                (uint32_t)*drained_w,
                (uint32_t)cw[tx_pub_total(lane)],
                total_pulled,
                (uint32_t)cw[tx_cls_pub(lane, 0)],
                cls_pulled[0],
                (uint32_t)cw[tx_done(lane, 0)],
                (uint32_t)cw[tx_cls_pub(lane, 1)],
                cls_pulled[1],
                (uint32_t)cw[tx_done(lane, 1)],
                (uint32_t)cw[tx_credit(kb, 0)],
                sent[kb][0]);
        }
#endif
        // close classes in order (END buffers go through the ring behind all class-c data of this lane)
        while (closed < G) {
            const uint32_t c = closed;
            const uint32_t producers = (c == 0 ? P : n_workers) / lanes3;
            if (cw[tx_done(lane, c)] < producers || cw[tx_cls_pub(lane, c)] != cls_pulled[c]) {
                break;
            }
#ifdef KVD_STATS
            if (c < 8 && tl_ready[c] == 0) {
                tl_ready[c] = stats_clock() / 135;
            }
#endif
            while (end_sent < ke && filled - *drained_w < kRing &&
                   sent[end_sent][c] - cw[tx_credit(end_sent, c)] < pool_depth(c, pool)) {
                const uint32_t i = filled % kRing;
                auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_l1(i));
                h[kHdrEntries] = 0;
                h[kHdrClass] = c;
                h[kHdrEnd] = 1;
                reserve(i, end_sent, c);
                stamp(i, 0, true);
                desc[i * 4 + 3] = 64;
                hand_over(1);
                issued = filled;  // (an END buffer has no read: it is handed over as it is chosen)
                ++end_sent;
            }
            if (end_sent < ke) {
                break;
            }
            end_sent = kb;
#ifdef KVD_STATS
            if (c < 8) {
                tl_closed[c] = stats_clock() / 135;
            }
#endif
            ++closed;
        }
        if (closed == G) {
            bool all_back = true;
            for (uint32_t k = kb; k < ke && all_back; ++k) {
                for (uint32_t c = 0; c < G; ++c) {
                    if (cw[tx_credit(k, c)] != sent[k][c]) {
                        all_back = false;
                        break;
                    }
                }
            }
            if (all_back) {
                break;
            }
        }
    }
    noc_async_read_set_trid(0);  // (the read command buffer's id outlives the kernel: leave it at the default)
    noc_async_atomic_barrier();  // the last frees have landed on their producers
    if constexpr (push) {
        // the writer has sent every buffer of ring B (it raises kHDrained after its sends left L1)
        while (*drained_w != filled) {
        }
        noc_semaphore_inc(get_noc_addr(tx_x, tx_y, ctrl + kTxRdoneB * 4), 1);
        noc_async_atomic_barrier();
    } else {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rdone)) = 1;
    }
#ifdef KVD_STATS
    if (d == 0 && l == 0) {
        DPRINT(
            "KVD3RC {} {} {} {} {} {} {} {} | {} {} {} {} {} {} {} {}\n",
            tl_ready[0],
            tl_ready[1],
            tl_ready[2],
            tl_ready[3],
            tl_ready[4],
            tl_ready[5],
            tl_ready[6],
            tl_ready[7],
            tl_closed[0],
            tl_closed[1],
            tl_closed[2],
            tl_closed[3],
            tl_closed[4],
            tl_closed[5],
            tl_closed[6],
            tl_closed[7]);
    }
    DPRINT("KVD3RS d={} l={} lane={} tries={} scans={}\n", d, l, lane, n_try, n_scan);
    DPRINT(
        "KVD3RL d={} l={} samples={} backlog_frac_pct={} backlog_avg_x10={} inflight_avg_x10={} ring_full_pct={} "
        "cap={} c0_backlog_avg_x10={} c0_inflight_avg_x10={} c0_cap={} c0_capped_pct={}\n",
        d,
        l,
        n_samp,
        n_samp ? 100 * n_samp_backlog / n_samp : 0,
        n_samp ? 10 * sum_backlog / n_samp : 0,
        n_samp ? 10 * sum_inflight / n_samp : 0,
        n_samp ? 100 * n_samp_full / n_samp : 0,
        (ke - kb) * ((G - 1) * pool + pool_depth(0, pool)),
        n_samp ? 10 * sum_backlog0 / n_samp : 0,
        n_samp ? 10 * sum_inflight0 / n_samp : 0,
        (ke - kb) * pool_depth(0, pool),
        n_samp ? 100 * n_samp_c0_capped / n_samp : 0);
    DPRINT(
        "KVD3R lane={} d={} l={} total={} choose={} bar={} hand={} none={} full={} batches={} pkts={} ok={} nok={} "
        "fail={} nfail={} pick={} rd={} tail={} empty={} nocredit={}\n",
        lane,
        d,
        l,
        stats_clock() - t0,
        cy_choose,
        cy_bar,
        cy_hand,
        cy_none,
        cy_full,
        n_batches,
        n_pk,
        cy_ok,
        n_ok,
        cy_fail,
        n_fail,
        cy_pick,
        cy_rd,
        cy_tail,
        n_why_empty,
        n_why_credit);
#endif
    (void)t0;
    (void)cy_choose;
    (void)n_try;
    (void)n_scan;
    (void)cy_bar;
    (void)cy_hand;
    (void)cy_none;
    (void)cy_full;
    (void)n_batches;
    (void)n_pk;
    (void)cy_ok;
    (void)cy_pick;
    (void)cy_rd;
    (void)cy_tail;
    (void)cy_fail;
    (void)n_ok;
    (void)n_fail;
    (void)n_why_empty;
    (void)n_why_credit;
}
