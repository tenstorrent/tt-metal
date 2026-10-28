// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// dynamic_claim_v2 — worker WRITER (BRISC / NoC1). Static share first (the op's store_block loop), then
// the pool blocks announced by the reader through cb_idx ({start_tile, valid}; SENTINEL ends the stream).
// After every block it publishes its running block count to the local SEM_WRITTEN word (the reader's
// pacing gate). The count is monotonic and only used for pacing, never for correctness.
// CT: [cb_out, cb_idx, block_tiles, page_bytes, accessor...]   RT: [dst_addr, static_start, static_tiles]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

constexpr uint32_t SEM_WRITTEN = 3;
constexpr uint32_t SENTINEL = 0xFFFFFFFFu;

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t static_start = get_arg_val<uint32_t>(1);
    const uint32_t static_tiles = get_arg_val<uint32_t>(2);
    const auto output_accessor = TensorAccessor(output_args, dst_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* written = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_WRITTEN));

    uint32_t nwritten = 0;
    auto store_block = [&](uint32_t page0, uint32_t valid) {
        cb_wait_front(cb_out, block_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        for (uint32_t i = 0; i < valid; ++i) {
            noc_async_write_page(page0 + i, output_accessor, l1_read_addr);
            l1_read_addr += page_bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, block_tiles);
        *written = ++nwritten;
    };
    {
        MaybeDeviceZoneScope("writer_write");
        for (uint32_t done = 0; done < static_tiles; done += block_tiles) {
            const uint32_t remaining = static_tiles - done;
            store_block(static_start + done, remaining < block_tiles ? remaining : block_tiles);
        }
        for (;;) {
            cb_wait_front(cb_idx, 1);
            volatile tt_l1_ptr uint32_t* slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_idx));
            invalidate_l1_cache();
            const uint32_t page0 = slot[0];
            const uint32_t valid = slot[1];
            cb_pop_front(cb_idx, 1);
            if (page0 == SENTINEL) {
                break;
            }
            store_block(page0, valid);
        }
    }
    {
        MaybeDeviceZoneScope("writer_final_barrier");
        noc_async_write_barrier();
    }
}
