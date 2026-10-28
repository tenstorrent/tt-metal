// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// dynamic_claim_v2 — DISPATCHER (one dedicated Tensix core, NCRISC / NoC0). Hands out pool block ids.
//
// WHY THIS EXISTS (vs round 1's NoC fetch-and-add): the INCR_GET return value is written into the
// requester's L1 by the responding NIU, i.e. an NIU-sourced inline write — the transaction class BH
// firmware never issues to Tensix L1 (noc_fast_spoof_write_dw_inline: "inline writes and atomics require
// all 4 memory ports ..."). The claims-only probe (kernels/faa_probe.cpp) shows the returned values are
// lost / delivered to other cores' return words: 11/300 launches bad with a polled return word, while
// read-once-after-barrier was 0/600 -- the value path is timing-sensitive, so no protocol built on it is
// provably correct (evidence_faa_probe.log). This protocol
// uses only primitives every BH kernel relies on: non-returning noc_semaphore_inc, NoC writes
// (noc_inline_dw_write = spoofed L1-sourced write on BH), NoC reads, and local L1 polls.
//
// RAW NoC: bypasses no helper (there is no work-queue helper); the request/grant protocol is hand-rolled.
//
// Protocol (single authority: only this RISC ever chooses a block id, ids handed out strictly increasing):
//   0. The FIRST ROUND (one chunk per worker, same formula as below with grant index 0) is computed on the
//      host and passed as runtime args, so workers start without any message. If it already covers the
//      pool, the host sets dynamic=0 everywhere and this kernel returns immediately.
//   1. Zero the request area REQ[W] (4 B per worker, in cb_req; CB L1 is stale across launches) with a self
//      NoC read from MEM_ZEROS and wait for the read barrier (the zeros are in L1 once it returns).
//   2. Multicast READY=1 into every worker's READY semaphore (semaphores are re-initialised to 0 by the
//      dispatcher firmware for every launch). A worker never sends a request before it has seen
//      READY, so no request can land before step 1 completed (causal ordering).
//   3. Sweep: pending(w) = REQ[w] - served[w]; each pending request gets the next chunk of pool blocks,
//      grant = (first_block << 12) | count (count >= 1, so never 0), or END, written with one 4-byte NoC
//      write into worker semaphore GRANT[served[w] % P]. Guided self-scheduling: count =
//      clamp(ceil(remaining / (F * W)), 1, KMAX) -> big chunks early (few grants), single blocks at the end
//      (fine balance). Per slot the values a worker sees are strictly increasing (first_block increases,
//      END = 0xFFFFFFFF is max), so "value changed" is an unambiguous new-grant signal and a worker never
//      needs to clear the slot.
//      The 4-byte write uses a per-worker L1 source word (flush=false): worker w's word is only rewritten
//      for w's next grant, which needs a request from w, which w only sends after it saw this grant land.
//   4. Exit once every worker has been sent END; write barrier so all grants land before this core reports
//      done (the program cannot finish before they land).
//
// CT: [cb_req, P, W, F, KMAX, RAMP0 (0 = off)]
// RT: [pool_blocks, first_round_blocks, worker_xy[0..W) packed x | (y << 16), (stats_addr), grid_lo_xy, grid_hi_xy]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef DC_STATS
#define DC_STATS 0
#endif
#ifndef READY_MCAST
#define READY_MCAST 1
#endif
#ifndef SWEEP_UNROLL
#define SWEEP_UNROLL 8
#endif
#ifndef GRANT_FLUSH
#define GRANT_FLUSH 0
#endif
constexpr uint32_t END = 0xFFFFFFFFu;
constexpr uint32_t SEM_READY = 4;
#ifndef REQ_STRIDE_WORDS
#define REQ_STRIDE_WORDS 1  // 4 B per worker: atomics address 32-bit lanes; 4 workers share one 16 B L1 word
#endif

// Per-worker state in RISC-local memory (.bss), not L1: the sweep must not pay an L1 round trip per worker
// for its own bookkeeping (measured: L1-resident served[] made the grant round trip ~2.4 us).
constexpr uint32_t W_MAX = 128;
uint32_t g_served[W_MAX];
uint32_t g_xy[W_MAX];

void kernel_main() {
    constexpr uint32_t cb_req = get_compile_time_arg_val(0);
    constexpr uint32_t P = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t F = get_compile_time_arg_val(3);
    constexpr uint32_t KMAX = get_compile_time_arg_val(4);
    constexpr uint32_t RAMP0 = get_compile_time_arg_val(5);
    static_assert(KMAX >= 1 && KMAX < 4096 && F >= 1);
    const uint32_t pool_blocks = get_arg_val<uint32_t>(0);
    const uint32_t first_round = get_arg_val<uint32_t>(1);
    if (first_round >= pool_blocks) {
        return;  // the host-assigned first round covers everything; no worker will request
    }

    const uint32_t req_base = get_write_ptr(cb_req);
    volatile tt_l1_ptr uint32_t* req = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(req_base);
    static_assert(W <= W_MAX);
    // The per-worker 16 B grant source words live in L1 after REQ (NoC-readable sources for the grant writes).
    constexpr uint32_t REQ_BYTES = (W * REQ_STRIDE_WORDS * 4 + 15) & ~15u;
    uint32_t* served = g_served;
    for (uint32_t w = 0; w < W; ++w) {
        g_xy[w] = get_arg_val<uint32_t>(2 + w);
    }
    const uint32_t src_base = (req_base + REQ_BYTES + W * 4 + 15) & ~15u;

    MaybeDeviceZoneScope("dispatch");
    {
        // 1. zero REQ (NoC read of the hardware zeros region into our own L1)
        const uint64_t zeros = get_noc_addr(my_x[noc_index], my_y[noc_index], MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < REQ_BYTES; off += MEM_ZEROS_SIZE) {
            const uint32_t rem = REQ_BYTES - off;
            noc_async_read(zeros, req_base + off, rem < MEM_ZEROS_SIZE ? rem : MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }

    uint32_t next = first_round;
    uint32_t ended = 0;
    auto give = [&](uint32_t w) {
        const uint32_t xy = g_xy[w];
        const uint32_t s = served[w];
        uint32_t val;
        if (next < pool_blocks) {
            const uint32_t rem = pool_blocks - next;
            uint32_t cnt = (rem + F * W - 1) / (F * W);
            cnt = cnt > KMAX ? KMAX : cnt;
            if constexpr (RAMP0 > 0) {
                // Rate-adaptive ramp: a worker's k-th grant is at most RAMP0 << k blocks. Slow cores (they come
                // back rarely) keep small chunks; fast cores ramp up to the guided size within a few grants.
                const uint32_t k = s + 1;  // grant index (the host-assigned first chunk is index 0)
                const uint32_t ramp = RAMP0 << (k < 12 ? k : 12);
                cnt = cnt > ramp ? ramp : cnt;
            }
            val = (next << 12) | cnt;
            next += cnt;
            served[w] = s + 1;
        } else {
            val = END;
            ++ended;
            served[w] = END;  // finished: later requests from w are ignored (w no longer waits)
        }
#if GRANT_FLUSH
        noc_inline_dw_write(get_noc_addr(xy & 0xFFFF, xy >> 16, get_semaphore(s % P)), val);
#else
        noc_inline_dw_write<InlineWriteDst::L1, false, false>(
            get_noc_addr(xy & 0xFFFF, xy >> 16, get_semaphore(s % P)),
            val,
            0xF,
            noc_index,
            NOC_UNICAST_WRITE_VC,
            src_base + w * 16);
#endif
    };

    // 2. READY to every worker (only after the zeroing above completed)
    for (uint32_t w = 0; w < W; ++w) {
        served[w] = 0;
    }
#if READY_MCAST
    {
        // One rectangle multicast over the whole worker grid (this core excluded: non-loopback variant).
        const uint32_t lo = get_arg_val<uint32_t>(2 + W + DC_STATS);
        const uint32_t hi = get_arg_val<uint32_t>(3 + W + DC_STATS);
        volatile tt_l1_ptr uint32_t* ready_src =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_READY));
        *ready_src = 1;
        noc_semaphore_set_multicast(
            get_semaphore(SEM_READY),
            get_noc_multicast_addr(lo & 0xFFFF, lo >> 16, hi & 0xFFFF, hi >> 16, get_semaphore(SEM_READY)),
            W);
    }
#else
    for (uint32_t w = 0; w < W; ++w) {
        const uint32_t xy = g_xy[w];
        noc_inline_dw_write(get_noc_addr(xy & 0xFFFF, xy >> 16, get_semaphore(SEM_READY)), 1);
    }
#endif
    // 3. serve requests until every worker has been told END
#if DC_STATS
    uint32_t sweeps = 0, grants = 0;
    const uint64_t t_s = get_timestamp();
#endif
#if DC_STATS
    uint32_t empty_sweeps = 0;
    uint64_t empty_t = 0;
#endif
    while (ended < W) {
#if DC_STATS
        ++sweeps;
        const uint64_t ts = get_timestamp();
        const uint32_t n0 = grants;
#endif
        invalidate_l1_cache();
        // Loads batched U at a time so their L1 latencies overlap (an uncached L1 load is ~15 cycles and a
        // load-compare-branch per worker measured ~18 cycles/worker = 2200 cycles per 119-worker sweep).
        constexpr uint32_t U = SWEEP_UNROLL;
        for (uint32_t w0 = 0; w0 < W; w0 += U) {
            uint32_t r[U];
#pragma GCC unroll 16
            for (uint32_t u = 0; u < U; ++u) {
                r[u] = (w0 + u < W) ? req[(w0 + u) * REQ_STRIDE_WORDS] : 0;
            }
#pragma GCC unroll 16
            for (uint32_t u = 0; u < U; ++u) {
                const uint32_t w = w0 + u;
                if (w < W) {
                    const uint32_t s = served[w];
                    if (s != END && r[u] > s) {
                        give(w);
#if DC_STATS
                        ++grants;
#endif
                    }
                }
            }
        }
#if DC_STATS
        if (grants == n0) {
            ++empty_sweeps;
            empty_t += get_timestamp() - ts;
        }
#endif
    }
    noc_async_write_barrier();
#if DC_STATS
    {
        constexpr auto stats_args = TensorAccessorArgs<6>();
        const uint32_t t = (uint32_t)(get_timestamp() - t_s);
        const auto stats = TensorAccessor(stats_args, get_arg_val<uint32_t>(2 + W), 64);
        volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_base + W * 16);
        st[0] = sweeps;
        st[1] = t;
        st[2] = grants;
        st[3] = empty_sweeps;
        st[4] = (uint32_t)empty_t;
        do {
            invalidate_l1_cache();
        } while (st[0] != sweeps || st[1] != t);
        noc_async_write(src_base + W * 16, stats.get_noc_addr(W), 64);
        noc_async_write_barrier();
    }
#endif
}
