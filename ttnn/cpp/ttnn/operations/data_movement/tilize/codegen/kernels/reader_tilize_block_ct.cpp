// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block tilize reader, uniform-width variant: reads a rectangular sub-block of RM sticks from DRAM.
//
// Same pipeline as reader_tilize_block.cpp but with num_col_chunks/chunk_Wt as compile-time args.
// Used when every core shares one block width (no width cliff, divisor column split): the constexpr
// width folds chunk_read_bytes and the stick-loop trip counts at compile time. Cliffed/ragged splits
// must keep reader_tilize_block.cpp's runtime ABI so mixed-width cores share one binary.
#include <stdint.h>
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    uint32_t start_stick = get_arg_val<uint32_t>(2);
    uint32_t col_byte_offset = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_col_chunks = get_compile_time_arg_val(0);  // sub-blocks per tile-row
    constexpr uint32_t H_per_tile = get_compile_time_arg_val(1);      // TILE_H (32)
    constexpr uint32_t chunk_Wt = get_compile_time_arg_val(2);        // tiles per sub-block
    constexpr uint32_t elem_w_bytes = get_compile_time_arg_val(3);    // TILE_W * elem_size
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr uint32_t chunk_read_bytes = chunk_Wt * elem_w_bytes;

    const auto s = TensorAccessor(src_args, src_addr, aligned_page_size);

    Noc noc;
    CircularBuffer cb_in_buf(cb_in);

    uint32_t i_stick = start_stick;

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        // Source byte offset within each stick of this tile-row. Every stick advances by
        // chunk_read_bytes per chunk, so a single running offset replaces the per-stick cached NOC
        // address array.
        uint32_t src_byte_offset = col_byte_offset;

        for (uint32_t c = 0; c < num_col_chunks; ++c) {
            cb_in_buf.reserve_back(chunk_Wt);
            // Hoist the CB write pointer out of the stick loop: a CB destination re-reads
            // fifo_wr_ptr from L1 on every async_read. The pointer cannot move between reserve_back
            // and push_back, so this is the same addresses in the same issue order.
            const CoreLocalMem<uint8_t> dst(cb_in_buf.get_write_ptr());
            uint32_t l1_write_offset = 0;
            for (uint32_t h = 0; h < H_per_tile; ++h) {
                noc.async_read(
                    s,
                    dst,
                    chunk_read_bytes,
                    {.page_id = i_stick + h, .offset_bytes = src_byte_offset},
                    {.offset_bytes = l1_write_offset});
                l1_write_offset += chunk_read_bytes;
            }
            src_byte_offset += chunk_read_bytes;
            noc.async_read_barrier();
            cb_in_buf.push_back(chunk_Wt);
        }

        i_stick += H_per_tile;
    }
}
