// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// work_rebalance experiment — STRIDED static ownership writer (mirror of reader_strided).
// CT: [cb_out, block_tiles, page_bytes, accessor...]  RT: [dst_addr, first_blk, stride, num_blocks, total_tiles]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_blk = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t total_tiles = get_arg_val<uint32_t>(4);
    const auto output_accessor = TensorAccessor(output_args, dst_addr, page_bytes);

    {
        MaybeDeviceZoneScope("writer_write");
        for (uint32_t blk = first_blk; blk < num_blocks; blk += stride) {
            const uint32_t page0 = blk * block_tiles;
            const uint32_t remaining = total_tiles - page0;
            const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
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
}
