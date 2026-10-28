// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// work_rebalance experiment — DYNAMIC work claiming reader (NCRISC / NoC0).
//
// KNOWN CORRECTNESS BLOCKER (not production-safe): on BH p150b the returned INCR_GET value is very
// occasionally wrong: a core gets a value that another core legitimately gets later, so one block is
// never written (see ../evidence_claimlog.log). Reproduced with block 2, block 8, and block 1 on an 8x8
// grid; not fixed by VC change, sentinel or monotonic polling, tagged idx slots, or a per-core unique
// return address. Root cause unknown.
//
// RAW NoC: bypasses the dataflow semaphore helpers (noc_semaphore_inc has no fetch-and-add return).
// Mechanism: noc_fast_atomic_increment<noc_mode, program_ret_addr=true> issues NOC_AT_INS_INCR_GET on the
// shared counter (semaphore SEM_COUNTER on the counter core); the NIU returns the PRE-increment value to
// our local L1 word (SEM_RET). After noc_async_atomic_barrier() (non-posted ack) the old value is our
// claimed chunk id. Fast cores claim more chunks -> work tracks throughput instead of being equal-static.
//
// Per claimed block the reader (1) pushes the block index into cb_idx for the writer, (2) bumps the
// local SEM_PUBLISHED word that the compute TRISCs poll, (3) streams the block into cb_input_tiles.
// On exhaustion it pushes SENTINEL to the writer and sets SEM_DONE for compute.
// The next claim is prefetched (issued before the current chunk's reads) to hide its round trip.
//
// CT: [cb_in, cb_idx, block_tiles, page_bytes, claim_blocks, accessor...]
// RT: [src_addr, counter_x, counter_y, num_blocks, total_tiles, stats_addr, core_id, guided_div, nonce]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef UNIQUE_RET
#define UNIQUE_RET 0
#endif
#ifndef ATOMIC_VC
#define ATOMIC_VC NOC_UNICAST_WRITE_VC
#endif
#ifndef CLAIM_GUIDED
#define CLAIM_GUIDED 0
#endif
#ifndef CLAIM_JIT
#define CLAIM_JIT 0
#endif
#ifndef WR_STATS
#define WR_STATS 0
#endif
constexpr uint32_t CB_STATS = 4;
constexpr uint32_t STATS_PAGE = 1024;
FORCE_INLINE void stats_log(uint32_t n, uint32_t v) {
#if WR_STATS
    if (n + 1 < STATS_PAGE / 4) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_STATS))[n + 1] = v;
    }
#endif
}

constexpr uint32_t SEM_COUNTER = 0;
constexpr uint32_t SEM_PUBLISHED = 1;
constexpr uint32_t SEM_DONE = 2;
constexpr uint32_t SEM_RET = 3;
constexpr uint32_t SENTINEL = 0xFFFFFFFFu;

// Fetch-and-add return protocol. Observed on BH: the non-posted atomic ack can retire BEFORE the
// returned value is visible in L1 (a plain read after noc_async_atomic_barrier() returned a stale value
// -> one chunk claimed twice, another never). An "arm the word with a sentinel, poll until overwritten"
// fix still hung ~1/1000 claims: the RISC arming store can land AFTER the NIU's return write under heavy
// L1 traffic. So NO RISC store touches the return word: the counter is monotonic and every claim adds
// >= 1, so this core's next returned value is strictly different from its previous one. SEM_RET starts
// at 0xFFFFFFFF (host initial value), which no claim can return. Poll until the word changes.
FORCE_INLINE uint32_t wait_claim(volatile tt_l1_ptr uint32_t* ret, uint32_t& prev) {
    noc_async_atomic_barrier();
    uint32_t v;
    do {
        invalidate_l1_cache();
        v = *ret;
    } while (v == prev);
    prev = v;
    return v;
}

// Slot = {start_tile, count, tag}; tag = nonce + push sequence number. The writer polls until the tag it
// expects is present, so it can never act on a stale slot (previous lap or previous launch).
uint32_t idx_tag;
FORCE_INLINE void push_idx(uint32_t cb_idx, uint32_t start_tile, uint32_t count) {
    cb_reserve_back(cb_idx, 1);
    volatile tt_l1_ptr uint32_t* slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_idx));
    slot[0] = start_tile;
    slot[1] = count;
    slot[2] = idx_tag++;
    // RISC L1 stores and the CB tiles_received register write take different paths; under heavy NoC
    // traffic into this L1 the push was observed to overtake the stores (writer read a stale slot ->
    // one block written twice, one never). Read back from L1 so the stores have landed before the push.
    do {
        invalidate_l1_cache();
    } while (slot[0] != start_tile || slot[1] != count || slot[2] != idx_tag - 1);
    cb_push_back(cb_idx, 1);
}

FORCE_INLINE void issue_claim(uint64_t counter_noc, uint32_t ret_addr, uint32_t incr = 1) {
    noc_fast_atomic_increment<noc_mode, true>(
        noc_index, write_at_cmd_buf, counter_noc, ATOMIC_VC, incr, 31, false, false, ret_addr);
}

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t claim_blocks = get_compile_time_arg_val(4);
    constexpr auto input_args = TensorAccessorArgs<5>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t counter_x = get_arg_val<uint32_t>(1);
    const uint32_t counter_y = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t total_tiles = get_arg_val<uint32_t>(4);

    const auto input_accessor = TensorAccessor(input_args, src_addr, page_bytes);
    const uint32_t num_chunks = (num_blocks + claim_blocks - 1) / claim_blocks;

    const uint64_t counter_noc = get_noc_addr(counter_x, counter_y, get_semaphore(SEM_COUNTER));
#if UNIQUE_RET
    // Diagnostic: per-core unique return offset (scratch CB 6, 16B per core). A return misdelivered to
    // another core then lands in an unused slot there instead of that core's live return word.
    const uint32_t ret_addr = get_write_ptr(6) + get_arg_val<uint32_t>(6) * 16;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ret_addr) = 0xFFFFFFFFu;
    do {
        invalidate_l1_cache();
    } while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ret_addr) != 0xFFFFFFFFu);
#else
    const uint32_t ret_addr = get_semaphore(SEM_RET);
#endif
    volatile tt_l1_ptr uint32_t* ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ret_addr);
    volatile tt_l1_ptr uint32_t* published =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_PUBLISHED));
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE));

    uint32_t my_blocks = 0;
    uint32_t ret_prev = 0xFFFFFFFFu;     // == SEM_RET initial value
    idx_tag = get_arg_val<uint32_t>(8);  // per-launch nonce
    {
        MaybeDeviceZoneScope("reader_read");
#if CLAIM_GUIDED
        // Guided self-scheduling (JIT): the shared counter counts TILES. Each claim takes
        // c = clamp((total - last_seen) / guided_div, 1, block_tiles) tiles, where last_seen is the counter
        // value this core observed on its previous claim: full blocks in steady state, shrinking blocks as
        // the work runs out, so the end-of-work in-flight amount on a slow core is small.
        // The CB quantum stays the nominal block_tiles (FullBlock); only `c` tiles are valid.
        const uint32_t guided_div = get_arg_val<uint32_t>(7);
        uint32_t last_seen = 0;
        for (;;) {
            cb_reserve_back(cb_in, block_tiles);
            const uint32_t rem = last_seen < total_tiles ? total_tiles - last_seen : 0;
            uint32_t c = rem / guided_div;
            c = c < 1 ? 1 : (c > block_tiles ? block_tiles : c);
            issue_claim(counter_noc, ret_addr, c);
            const uint32_t page0 = wait_claim(ret, ret_prev);
            last_seen = page0 + c;
            if (page0 >= total_tiles) {
                break;
            }
            const uint32_t left = total_tiles - page0;
            const uint32_t valid = left < c ? left : c;
            push_idx(cb_idx, page0, valid);
            stats_log(my_blocks, page0);
            ++my_blocks;
            *published = my_blocks;
            uint32_t l1_write_addr = get_write_ptr(cb_in);
            for (uint32_t i = 0; i < valid; ++i) {
                noc_async_read_page(page0 + i, input_accessor, l1_write_addr);
                l1_write_addr += page_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, block_tiles);
        }
#elif CLAIM_JIT
        // Just-in-time claim: only claim once input space is reserved, so a slow core never sits on a
        // claimed-but-unstarted block (shrinks the end-of-work tail). Round trip is exposed per block.
        static_assert(claim_blocks == 1, "CLAIM_JIT claims one block at a time");
        for (;;) {
            cb_reserve_back(cb_in, block_tiles);
            issue_claim(counter_noc, ret_addr);
            const uint32_t blk = wait_claim(ret, ret_prev);
            if (blk >= num_blocks) {
                break;
            }
            const uint32_t page0 = blk * block_tiles;
            const uint32_t remaining = total_tiles - page0;
            const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
            push_idx(cb_idx, page0, valid);
            stats_log(my_blocks, page0);
            ++my_blocks;
            *published = my_blocks;
            uint32_t l1_write_addr = get_write_ptr(cb_in);
            for (uint32_t i = 0; i < valid; ++i) {
                noc_async_read_page(page0 + i, input_accessor, l1_write_addr);
                l1_write_addr += page_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, block_tiles);
        }
#else
        issue_claim(counter_noc, ret_addr);
        uint32_t chunk = wait_claim(ret, ret_prev);
        while (chunk < num_chunks) {
            issue_claim(counter_noc, ret_addr);  // prefetch the next claim
            const uint32_t blk_begin = chunk * claim_blocks;
            const uint32_t blk_end = blk_begin + claim_blocks < num_blocks ? blk_begin + claim_blocks : num_blocks;
            for (uint32_t blk = blk_begin; blk < blk_end; ++blk) {
                const uint32_t page0 = blk * block_tiles;
                const uint32_t remaining = total_tiles - page0;
                const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
                push_idx(cb_idx, page0, valid);
                stats_log(my_blocks, page0);
                ++my_blocks;
                *published = my_blocks;
                cb_reserve_back(cb_in, block_tiles);
                uint32_t l1_write_addr = get_write_ptr(cb_in);
                for (uint32_t i = 0; i < valid; ++i) {
                    noc_async_read_page(page0 + i, input_accessor, l1_write_addr);
                    l1_write_addr += page_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_in, block_tiles);
            }
            chunk = wait_claim(ret, ret_prev);
        }
#endif
        push_idx(cb_idx, SENTINEL, 0);
        *done = 1;
    }
    // Restore the default atomic return address so later programs' atomics don't land in our L1 word.
    NOC_CMD_BUF_WRITE_REG(noc_index, write_at_cmd_buf, NOC_RET_ADDR_LO, MEM_NOC_ATOMIC_RET_VAL_ADDR);

#if WR_STATS
    {
        // Stats (debug only): 1KB ROW_MAJOR page per core: [count, start_tile of each claim...].
        const uint32_t stats_addr = get_arg_val<uint32_t>(5);
        const uint32_t core_id = get_arg_val<uint32_t>(6);
        constexpr auto stats_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
        const auto stats_accessor = TensorAccessor(stats_args, stats_addr, STATS_PAGE);
        volatile tt_l1_ptr uint32_t* log = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_STATS));
        log[0] = my_blocks;
        invalidate_l1_cache();
        while (log[0] != my_blocks) {
            invalidate_l1_cache();
        }
        noc_async_write(get_write_ptr(CB_STATS), stats_accessor.get_noc_addr(core_id), STATS_PAGE);
        noc_async_write_barrier();
    }
#endif
}
