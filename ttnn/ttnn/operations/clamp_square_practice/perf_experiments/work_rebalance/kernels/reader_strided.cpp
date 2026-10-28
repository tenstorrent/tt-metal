// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// work_rebalance experiment — STRIDED static ownership reader: core c owns blocks c, c+N, c+2N, ...
// (block = block_tiles contiguous pages). Same per-block read path as the op's reader.
// CT: [cb_in, block_tiles, page_bytes, accessor...]  RT: [src_addr, first_blk, stride, num_blocks, total_tiles]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto input_args = TensorAccessorArgs<3>();
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_blk = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t total_tiles = get_arg_val<uint32_t>(4);
    const auto input_accessor = TensorAccessor(input_args, src_addr, page_bytes);

    MaybeDeviceZoneScope("reader_read");
    for (uint32_t blk = first_blk; blk < num_blocks; blk += stride) {
        const uint32_t page0 = blk * block_tiles;
        const uint32_t remaining = total_tiles - page0;
        const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
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
