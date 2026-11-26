// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast reader (NCRISC / NoC0) — load_block.
//
// Streams this core's page range of the interleaved input (DRAM or L1) into cb_input_tiles,
// block_tiles pages per handshake and one read barrier per block. Raw dataflow API: no kernel_lib
// helper reads interleaved tile pages (op_design.md, "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t first_page = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);
    const uint32_t page_stride = get_arg_val<uint32_t>(2);

    const auto src = TensorAccessor(src_args, src_addr, page_bytes);

    // Every non-final handshake is exactly block_tiles; the ragged block is the last one on the
    // CB, so the ring pointer never has to wrap mid-block.
    const uint32_t num_blocks = (core_tiles + block_tiles - 1) / block_tiles;
    uint32_t page = first_page;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_extent =
            (block_idx + 1 == num_blocks) ? core_tiles - block_idx * block_tiles : block_tiles;
        {
            DeviceZoneScopedN("TC-RD-WAIT");
            cb_reserve_back(cb_input_tiles, block_extent);
        }
        {
            DeviceZoneScopedN("TC-RD-ISSUE");
            uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
            for (uint32_t j = 0; j < block_extent; ++j) {
                noc_async_read(src.get_noc_addr(page), l1_write_addr, page_bytes);
                l1_write_addr += page_bytes;
                page += page_stride;
            }
        }
        {
            DeviceZoneScopedN("TC-RD-BARRIER");
            noc_async_read_barrier();
        }
        cb_push_back(cb_input_tiles, block_extent);
    }
}
