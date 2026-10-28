// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// dynamic_claim_v2 — worker READER (NCRISC / NoC0).
// Phase 1: this worker's FIRST CHUNK (host-computed, exactly the dispatcher's first-round formula, so no
//          message is needed to start) streamed with the op's load_block loop.
// Phase 2 (only if the host says pool blocks remain after the first round): pool chunks granted by the
//          dispatcher core (protocol in dispatcher.cpp).
//
// Request side: one non-returning noc_semaphore_inc(+1) on REQ[w] in the dispatcher's L1 per request, and
// NEVER before the local READY semaphore is set: the dispatcher sets READY only after REQ[] is zeroed (NoC
// read barrier), so no request can be wiped by the zeroing (causal order: zero -> READY write -> READY seen
// -> request). Grants ((first_block << 12) | count, or END) arrive in local semaphores GRANT[k % P]
// (k = dynamic grant sequence number); per slot values strictly increase, so a new grant is "slot value !=
// last value seen in that slot" and slots never need clearing.
// Writer pacing: a request is only sent while (blocks obtained + outstanding requests) - blocks written
// <= AHEAD, where "blocks written" is the writer's local SEM_WRITTEN counter. Without it every reader would
// drain the pool at READ speed (reads are ~2x faster than writes and the CBs hold a whole share), which
// balances nothing; with it, pool blocks follow each core's WRITE throughput (the bottleneck).
// Hand-off: each pool block's {start_tile, valid} goes to the writer through cb_idx; the dynamic block
// count goes to compute through the SEM_PUB word = (count << 1) | done (one 32-bit store, monotonic).
//
// RAW NoC: bypasses no helper (no work-queue helper exists); see dispatcher.cpp for why not fetch-and-add.
//
// CT: [cb_in, cb_idx, cb_req, block_tiles, page_bytes, P, AHEAD, accessor...]
// RT: [src_addr, first_start_tile, first_tiles, pool_start, total_tiles, disp_xy (x | y << 16), worker_id,
//      dynamic (0|1), (stats_addr)]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef DC_STATS
#define DC_STATS 0  // diagnostics: per-worker {grant_wait, gate_wait, grants, blocks, t_end} to a DRAM page
#endif
#ifndef REQ_STRIDE_WORDS
#define REQ_STRIDE_WORDS 1
#endif

constexpr uint32_t SEM_PUB = 2;
constexpr uint32_t SEM_WRITTEN = 3;
constexpr uint32_t SEM_READY = 4;
constexpr uint32_t END = 0xFFFFFFFFu;
constexpr uint32_t SENTINEL = 0xFFFFFFFFu;

FORCE_INLINE void push_idx(uint32_t cb_idx, uint32_t start_tile, uint32_t count) {
    cb_reserve_back(cb_idx, 1);
    volatile tt_l1_ptr uint32_t* slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_idx));
    slot[0] = start_tile;
    slot[1] = count;
    // read back so both stores are in L1 before the push becomes visible to the writer
    do {
        invalidate_l1_cache();
    } while (slot[0] != start_tile || slot[1] != count);
    cb_push_back(cb_idx, 1);
}

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t cb_req = get_compile_time_arg_val(2);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t P = get_compile_time_arg_val(5);
    constexpr uint32_t AHEAD = get_compile_time_arg_val(6);
    constexpr auto input_args = TensorAccessorArgs<7>();
    static_assert(P >= 1 && P <= 2 && AHEAD >= 1, "P in {1,2}; AHEAD >= 1 (else the pacing gate can deadlock)");

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t static_start = get_arg_val<uint32_t>(1);
    const uint32_t static_tiles = get_arg_val<uint32_t>(2);
    const uint32_t pool_start = get_arg_val<uint32_t>(3);
    const uint32_t total_tiles = get_arg_val<uint32_t>(4);
    const uint32_t disp_xy = get_arg_val<uint32_t>(5);
    const uint32_t worker_id = get_arg_val<uint32_t>(6);
    const uint32_t dynamic = get_arg_val<uint32_t>(7);

    const auto input_accessor = TensorAccessor(input_args, src_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* pub = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_PUB));
    volatile tt_l1_ptr uint32_t* written = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_WRITTEN));
    const uint64_t req_noc =
        get_noc_addr(disp_xy & 0xFFFF, disp_xy >> 16, get_write_ptr(cb_req) + worker_id * REQ_STRIDE_WORDS * 4);

    MaybeDeviceZoneScope("reader_read");
    // ---- first chunk (identical to the op's reader) ----
    for (uint32_t done = 0; done < static_tiles; done += block_tiles) {
        const uint32_t remaining = static_tiles - done;
        const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
        cb_reserve_back(cb_in, block_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_in);
        const uint32_t page0 = static_start + done;
        for (uint32_t i = 0; i < valid; ++i) {
            noc_async_read_page(page0 + i, input_accessor, l1_write_addr);
            l1_write_addr += page_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, block_tiles);
    }
    const uint32_t nstatic = (static_tiles + block_tiles - 1) / block_tiles;

    // ---- pool blocks ----
    uint32_t last[2] = {0, 0};  // GRANT semaphores start at 0 each launch
    uint32_t issued = 0;
    uint32_t consumed = 0;
    uint32_t ndyn = 0;
#if DC_STATS
    uint64_t t_grant = 0, t_gate = 0;
    const uint64_t t_start = get_timestamp();
#endif
    if (dynamic) {
        // Never request before READY (see header). A request sent before the dispatcher zeroed REQ[] would be
        // wiped -> lost request -> hang (observed under --dev in an earlier revision without this rule).
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_READY));
        do {
            invalidate_l1_cache();
        } while (*ready == 0);
    }
    while (dynamic) {
        // keep up to P requests outstanding, paced by the writer
        while (issued - consumed < P) {
            invalidate_l1_cache();
            const bool gate_open = nstatic + ndyn + (issued - consumed) <= *written + AHEAD;
            if (!gate_open) {
                if (issued != consumed) {
                    break;  // a grant is still coming; don't stall on the gate
                }
#if DC_STATS
                const uint64_t t0 = get_timestamp();
                do {
                    invalidate_l1_cache();
                } while (nstatic + ndyn > *written + AHEAD);
                t_gate += get_timestamp() - t0;
#endif
                continue;  // nothing outstanding: wait for the writer to drain
            }
            noc_semaphore_inc(req_noc, 1);
            ++issued;
        }
        const uint32_t slot = consumed % P;
        volatile tt_l1_ptr uint32_t* g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(slot));
        uint32_t v;
#if DC_STATS
        const uint64_t t0 = get_timestamp();
#endif
        do {
            invalidate_l1_cache();
            v = *g;
        } while (v == last[slot]);
#if DC_STATS
        t_grant += get_timestamp() - t0;
#endif
        last[slot] = v;
        ++consumed;
        if (v == END) {
            break;
        }
        const uint32_t first = v >> 12;
        const uint32_t cnt = v & 0xFFFu;
        for (uint32_t b = 0; b < cnt; ++b) {
            const uint32_t page0 = pool_start + (first + b) * block_tiles;
            const uint32_t left = total_tiles - page0;
            const uint32_t valid = left < block_tiles ? left : block_tiles;
            push_idx(cb_idx, page0, valid);
            ++ndyn;
            *pub = ndyn << 1;
            cb_reserve_back(cb_in, block_tiles);
            uint32_t l1_write_addr = get_write_ptr(cb_in);
            for (uint32_t i = 0; i < valid; ++i) {
                noc_async_read_page(page0 + i, input_accessor, l1_write_addr);
                l1_write_addr += page_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, block_tiles);
        }
    }
    push_idx(cb_idx, SENTINEL, 0);
    *pub = (ndyn << 1) | 1u;
    noc_async_atomic_barrier();  // every request inc has landed before this core reports done
#if DC_STATS
    {
        constexpr auto stats_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
        const auto stats = TensorAccessor(stats_args, get_arg_val<uint32_t>(8), 64);
        constexpr uint32_t CB_STATS = 4;
        volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_STATS));
        st[0] = (uint32_t)t_grant;
        st[1] = (uint32_t)t_gate;
        st[2] = consumed;
        st[3] = ndyn;
        st[4] = (uint32_t)(get_timestamp() - t_start);
        do {
            invalidate_l1_cache();
        } while (st[3] != ndyn);
        noc_async_write(get_write_ptr(CB_STATS), stats.get_noc_addr(worker_id), 64);
        noc_async_write_barrier();
    }
#endif
}
