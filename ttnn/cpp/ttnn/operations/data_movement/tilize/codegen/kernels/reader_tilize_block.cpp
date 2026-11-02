// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block tilize reader: reads a rectangular sub-block of RM sticks from DRAM.
//
// Each core handles a (num_tile_rows x block_Wt) region of the tensor.
// Reads partial sticks starting at col_byte_offset within each stick.
// Supports sub-block column chunking when block_Wt exceeds L1.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TILIZE_CAPPED_BLOCKS
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/page.h"
#endif

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

#ifdef TILIZE_CAPPED_BLOCKS
    // Capped column blocks: every slab is pushed at the nominal chunk_Wt so
    // each CB reservation stays a whole slab of a 2-slab FIFO; the final slab
    // of a row reads only its last_Wt valid tiles. The L1 stick pitch stays
    // nominal because tilize_block reads a chunk_Wt-wide slab.
    const uint32_t last_read_bytes = get_arg_val<uint32_t>(4) * elem_w_bytes;
    const uint8_t noc_id = noc.get_noc_id();
    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        // TensorAccessor adds the byte offset after resolving the page, so
        // the per-stick NoC base is resolved once per tile row.
        uint64_t base_noc[H_per_tile];
        for (uint32_t h = 0; h < H_per_tile; ++h) {
            base_noc[h] = s.get_noc_addr(i_stick + h, 0, noc_id);
        }
        uint32_t src_byte_offset = col_byte_offset;
        for (uint32_t c = 0; c < num_col_chunks; ++c) {
            const uint32_t read_bytes = c + 1 == num_col_chunks ? last_read_bytes : chunk_read_bytes;
            cb_in_buf.reserve_back(chunk_Wt);
            const CoreLocalMem<uint8_t> dst(cb_in_buf.get_write_ptr());
            uint32_t l1_write_offset = 0;
            for (uint32_t h = 0; h < H_per_tile; ++h) {
                noc.async_read(
                    tensor_accessor::Page(base_noc[h], i_stick + h),
                    dst,
                    read_bytes,
                    {.offset_bytes = src_byte_offset},
                    {.offset_bytes = l1_write_offset});
                l1_write_offset += chunk_read_bytes;
            }
            src_byte_offset += chunk_read_bytes;
            noc.async_read_barrier();
            cb_in_buf.push_back(chunk_Wt);
        }
        i_stick += H_per_tile;
    }
#else
    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        // Source byte offset within each stick of this tile-row. Every stick
        // advances by chunk_read_bytes per chunk, so a single running offset
        // replaces the per-stick cached NOC address array.
        uint32_t src_byte_offset = col_byte_offset;

        for (uint32_t c = 0; c < num_col_chunks; ++c) {
            cb_in_buf.reserve_back(chunk_Wt);
            uint32_t l1_write_offset = 0;
            for (uint32_t h = 0; h < H_per_tile; ++h) {
                noc.async_read(
                    s,
                    cb_in_buf,
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
#endif
}
