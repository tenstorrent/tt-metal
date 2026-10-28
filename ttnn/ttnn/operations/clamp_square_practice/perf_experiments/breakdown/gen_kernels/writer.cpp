// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// clamp_square_practice writer (BRISC / NoC1) — store_block.
// Drains cb_output_tiles in blocks of block_tiles (nominal quantum, FullBlock contract) and
// writes the `valid` pages of each block to the same page indices as the input: one wait,
// `valid` async writes, one flush, one pop per block; one final write barrier per kernel.

#include <cstdint>
#ifndef ABLATE_WRITE
#define ABLATE_WRITE 0
#endif

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_tiles = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_args, dst_addr, page_bytes);

    // Permanent stage zones (free when the profiler is off). Kernel-level, never per block.
    // writer_write is occupancy: includes cb_wait_front stalls on compute.
    {
        MaybeDeviceZoneScope("writer_write");
        for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
            const uint32_t remaining = core_tiles - done;
            const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
            cb_wait_front(cb_output_tiles, block_tiles);
            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
            const uint32_t page0 = start_tile + done;
            for (uint32_t i = 0; i < valid; ++i) {
                if constexpr (!ABLATE_WRITE) {
                    noc_async_write_page(page0 + i, output_accessor, l1_read_addr);
                }
                l1_read_addr += page_bytes;
            }
            noc_async_writes_flushed();  // L1 source reads done -> slot can be recycled
            cb_pop_front(cb_output_tiles, block_tiles);
        }
    }
    {
        MaybeDeviceZoneScope("writer_final_barrier");
        noc_async_write_barrier();
    }
}
