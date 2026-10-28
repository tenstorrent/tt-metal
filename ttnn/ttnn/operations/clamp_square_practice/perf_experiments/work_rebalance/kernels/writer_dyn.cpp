// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// work_rebalance experiment — DYNAMIC writer (BRISC / NoC1).
// Blocks arrive through cb_idx as {start_tile, valid_count} (one 16B page per block, pushed by the
// reader at claim time);
// SENTINEL ends the stream. Write path per block is identical to the op's writer.
// CT: [cb_out, cb_idx, block_tiles, page_bytes, accessor...]   RT: [dst_addr, total_tiles, stats_addr, stats_row,
// nonce]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef WR_STATS
#define WR_STATS 0
#endif

constexpr uint32_t SENTINEL = 0xFFFFFFFFu;

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t total_tiles = get_arg_val<uint32_t>(1);
    const auto output_accessor = TensorAccessor(output_args, dst_addr, page_bytes);

    uint32_t wcount = 0;
    uint32_t tag = get_arg_val<uint32_t>(4);  // per-launch nonce (same as reader's)
    {
        MaybeDeviceZoneScope("writer_write");
        for (;;) {
            cb_wait_front(cb_idx, 1);
            volatile tt_l1_ptr uint32_t* slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_idx));
            do {
                invalidate_l1_cache();
            } while (slot[2] != tag);
            ++tag;
            const uint32_t page0 = slot[0];
            const uint32_t valid = slot[1];
            cb_pop_front(cb_idx, 1);
            if (page0 == SENTINEL) {
                break;
            }
#if WR_STATS
            if (wcount + 1 < 256) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(5))[wcount + 1] = page0;
            }
#endif
            ++wcount;
            cb_wait_front(cb_out, block_tiles);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            for (uint32_t i = 0; i < valid; ++i) {
                noc_async_write_page(page0 + i, output_accessor, l1_read_addr);
                l1_read_addr += page_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, block_tiles);
        }
    }
    {
        MaybeDeviceZoneScope("writer_final_barrier");
        noc_async_write_barrier();
    }
#if WR_STATS
    {
        // Stats (debug only): 1KB page per core at row (grid_cores + core_id): [count, page0 written...]
        constexpr uint32_t CB_WSTATS = 5;
        constexpr uint32_t STATS_PAGE = 1024;
        const uint32_t stats_addr = get_arg_val<uint32_t>(2);
        const uint32_t row = get_arg_val<uint32_t>(3);
        constexpr auto stats_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
        const auto stats_accessor = TensorAccessor(stats_args, stats_addr, STATS_PAGE);
        volatile tt_l1_ptr uint32_t* log = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_WSTATS));
        log[0] = wcount;
        invalidate_l1_cache();
        while (log[0] != wcount) {
            invalidate_l1_cache();
        }
        noc_async_write(get_write_ptr(CB_WSTATS), stats_accessor.get_noc_addr(row), STATS_PAGE);
        noc_async_write_barrier();
    }
#endif
}
