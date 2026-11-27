// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast reader (NCRISC / NoC0): load_block.
//
// Walks this core's page range of the interleaved input (DRAM or L1, resolved by TensorAccessor)
// in blocks of `block_tiles` pages: one reserve, `block_extent` async reads, one read barrier and
// one push per block. Raw dataflow API: no kernel_lib helper reads interleaved tile pages.
//
// Ring invariant: cb_input_tiles holds IN_DEPTH * block_tiles pages, every non-final handshake is
// exactly block_tiles, and the ragged block is the last handshake of the program on this CB, so
// the write pointer only ever wraps on an exact fifo_limit hit.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t in_page_bytes = get_compile_time_arg_val(2);
    constexpr auto input_args = TensorAccessorArgs<3>();

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t first_page = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);
    const uint32_t page_stride = get_arg_val<uint32_t>(2);

    const auto input = TensorAccessor(input_args, input_addr, in_page_bytes);

    uint32_t page = first_page;
    for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
        const uint32_t block_extent = (core_tiles - done) < block_tiles ? (core_tiles - done) : block_tiles;
        {
            DeviceZoneScopedN("TC-RD-WAIT");
            cb_reserve_back(cb_input_tiles, block_extent);
        }
        {
            DeviceZoneScopedN("TC-RD-ISSUE");
            uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
            for (uint32_t j = 0; j < block_extent; ++j) {
                noc_async_read(input.get_noc_addr(page), l1_write_addr, in_page_bytes);
                l1_write_addr += in_page_bytes;
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
