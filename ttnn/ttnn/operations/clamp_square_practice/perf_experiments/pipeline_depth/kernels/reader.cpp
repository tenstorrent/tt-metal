// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// pipeline_depth experiment reader (NCRISC / NoC0).
// Same as the op's reader, except the DM transfer block `dm_tiles` (reads per barrier / per
// push) is DECOUPLED from the compute DEST block `cb_block`. dm_tiles is a multiple of cb_block.
// Ragged tail keeps the compute FullBlock contract: the last chunk pushes round_up(valid,
// cb_block) pages, so the total pushed == ceil(core_tiles / cb_block) * cb_block, exactly what
// the FullBlock compute chain waits for. CB size is a multiple of dm_tiles, so every reserve
// is contiguous (no straddled wrap). No raw LLK; plain dataflow API.
//
// CT args: [cb_input_tiles, dm_tiles, page_bytes, cb_block, TensorAccessorArgs...]
// RT args: [src_addr, start_tile, core_tiles]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t dm_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_block = get_compile_time_arg_val(3);
    constexpr auto input_args = TensorAccessorArgs<4>();
    static_assert(dm_tiles % cb_block == 0, "DM block must be a multiple of the compute block");

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_tiles = get_arg_val<uint32_t>(2);

    const auto input_accessor = TensorAccessor(input_args, src_addr, page_bytes);

    MaybeDeviceZoneScope("reader_read");
    for (uint32_t done = 0; done < core_tiles; done += dm_tiles) {
        const uint32_t remaining = core_tiles - done;
        const bool last = remaining <= dm_tiles;
        const uint32_t valid = last ? remaining : dm_tiles;
        const uint32_t quantum = last ? ((valid + cb_block - 1) / cb_block) * cb_block : dm_tiles;
        cb_reserve_back(cb_input_tiles, quantum);
        uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
        const uint32_t page0 = start_tile + done;
        for (uint32_t i = 0; i < valid; ++i) {
            noc_async_read_page(page0 + i, input_accessor, l1_write_addr);
            l1_write_addr += page_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, quantum);
    }
}
