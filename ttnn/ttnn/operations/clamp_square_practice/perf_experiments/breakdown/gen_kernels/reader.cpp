// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// clamp_square_practice reader (NCRISC / NoC0) — load_block.
// Streams this core's contiguous page range [start_tile, start_tile + core_tiles) into
// cb_input_tiles in blocks of block_tiles: one reserve, `valid` async reads, ONE barrier,
// one push per block. The CB quantum is always the nominal block_tiles (FullBlock contract
// with the compute chain), including the ragged last block.

#include <cstdint>
#ifndef ABLATE_READ
#define ABLATE_READ 0
#endif

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto input_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_tiles = get_arg_val<uint32_t>(2);

    const auto input_accessor = TensorAccessor(input_args, src_addr, page_bytes);

    // Permanent stage zone (free when the profiler is off). Kernel-level, never per block: the
    // per-RISC marker budget is exhausted silently. Occupancy: includes cb_reserve_back stalls.
    MaybeDeviceZoneScope("reader_read");
    for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
        const uint32_t remaining = core_tiles - done;
        const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
        cb_reserve_back(cb_input_tiles, block_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
        const uint32_t page0 = start_tile + done;
        for (uint32_t i = 0; i < valid; ++i) {
            if constexpr (!ABLATE_READ) {
                noc_async_read_page(page0 + i, input_accessor, l1_write_addr);
            }
            l1_write_addr += page_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, block_tiles);
    }
}
