// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// pipeline_depth experiment writer (BRISC / NoC1).
// Same as the op's writer, except the DM drain block `dm_tiles` (writes per flush / per pop) is
// DECOUPLED from the compute DEST block `cb_block` (dm_tiles % cb_block == 0). The compute chain
// pushes cb_block at a time; this kernel waits for dm_tiles of them. Ragged tail: the last chunk
// waits/pops round_up(valid, cb_block) (FullBlock contract), writes only `valid` pages.
//
// CT args: [cb_output_tiles, dm_tiles, page_bytes, cb_block, TensorAccessorArgs...]
// RT args: [dst_addr, start_tile, core_tiles]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t dm_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_block = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();
    static_assert(dm_tiles % cb_block == 0, "DM block must be a multiple of the compute block");

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_tiles = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_args, dst_addr, page_bytes);

    {
        MaybeDeviceZoneScope("writer_write");
        for (uint32_t done = 0; done < core_tiles; done += dm_tiles) {
            const uint32_t remaining = core_tiles - done;
            const bool last = remaining <= dm_tiles;
            const uint32_t valid = last ? remaining : dm_tiles;
            const uint32_t quantum = last ? ((valid + cb_block - 1) / cb_block) * cb_block : dm_tiles;
            cb_wait_front(cb_output_tiles, quantum);
            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
            const uint32_t page0 = start_tile + done;
            for (uint32_t i = 0; i < valid; ++i) {
                noc_async_write_page(page0 + i, output_accessor, l1_read_addr);
                l1_read_addr += page_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_output_tiles, quantum);
        }
    }
    {
        MaybeDeviceZoneScope("writer_final_barrier");
        noc_async_write_barrier();
    }
}
