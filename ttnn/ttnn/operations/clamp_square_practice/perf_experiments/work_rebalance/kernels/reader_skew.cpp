// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// work_rebalance experiment: op reader + per-core rotated in-block page order (RT arg 3 = rot).
// Page j still lands in slot j, so compute/writer order is unchanged; only the DRAM bank hit
// order at a given instant differs between cores (de-synchronizes bank access).
// clamp_square_practice reader (NCRISC / NoC0) — load_block.
// Streams this core's contiguous page range [start_tile, start_tile + core_tiles) into
// cb_input_tiles in blocks of block_tiles: one reserve, `valid` async reads, ONE barrier,
// one push per block. The CB quantum is always the nominal block_tiles (FullBlock contract
// with the compute chain), including the ragged last block.

#include <cstdint>

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
    const uint32_t rot = get_arg_val<uint32_t>(3);  // work_rebalance: per-core in-block start skew

    const auto input_accessor = TensorAccessor(input_args, src_addr, page_bytes);

    // Permanent stage zone (free when the profiler is off). Kernel-level, never per block: the
    // per-RISC marker budget is exhausted silently. Occupancy: includes cb_reserve_back stalls.
    MaybeDeviceZoneScope("reader_read");
    for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
        const uint32_t remaining = core_tiles - done;
        const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
        cb_reserve_back(cb_input_tiles, block_tiles);
        const uint32_t l1_base = get_write_ptr(cb_input_tiles);
        const uint32_t page0 = start_tile + done;
        uint32_t j = rot % valid;
        for (uint32_t i = 0; i < valid; ++i) {
            noc_async_read_page(page0 + j, input_accessor, l1_base + j * page_bytes);
            j = (j + 1 == valid) ? 0 : j + 1;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, block_tiles);
    }
}
