// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax writer kernel — V2 streaming path.
//
// The V2 compute kernel writes output tiles one chunk at a time. The writer
// drains them as they arrive.
//
// TILE path: reads tiles from cb_output_tiles, writes to DRAM/L1
//   For dim=-1: tiles arrive in row-major order (standard)
//   For dim=-2: tiles arrive in column-major order (per-column chunks)
//
// RM path: reads tile-pages from cb_rm_out (compute untilizes), writes to DRAM/L1
//   Uses write_sticks_after_untilize with byte_offset_within_page to write
//   W-slices of each stick.
//
// write_sticks_after_untilize calls cb_wait_front / cb_pop_front internally
// (width_in_tiles pages per block of 32 rows). The writer just calls it.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_rm_out = 17;
}  // namespace

void kernel_main() {
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_slabs = get_arg_val<uint32_t>(2);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t is_rm = get_compile_time_arg_val(2);
    constexpr uint32_t origin_W = get_compile_time_arg_val(3);
    constexpr uint32_t origin_H = get_compile_time_arg_val(4);
    constexpr int32_t dim = static_cast<int32_t>(get_compile_time_arg_val(5));
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(6);
    constexpr uint32_t chunk_along_reduce = get_compile_time_arg_val(7);

    constexpr auto dst_args = TensorAccessorArgs<8>();
    const auto dst_accessor = TensorAccessor(dst_args, output_buffer_address);

    constexpr uint32_t reduce_dim_tiles = (dim == -1) ? Wt : Ht;
    constexpr uint32_t non_reduce_dim = (dim == -1) ? Ht : Wt;
    constexpr uint32_t num_chunks =
        chunk_along_reduce ? (reduce_dim_tiles / BLOCK_SIZE) : (non_reduce_dim / BLOCK_SIZE);

    if constexpr (!is_rm) {
        // ===== TILE path: write tiles from cb_output_tiles =====
        CircularBuffer output_cb(cb_output_tiles);
        Noc noc;
        const uint32_t tile_bytes = get_tile_size(cb_output_tiles);

        uint32_t slab_start_tile = start_id;

        for (uint32_t slab = 0; slab < num_slabs; ++slab) {
            if constexpr (!chunk_along_reduce) {
                // chunk_along_non_reduce: tiles arrive chunked along the non-reduce dim
                if constexpr (dim == -1) {
                    // dim=-1: chunks along Ht. Each chunk: BLOCK_SIZE rows × Wt cols.
                    // Tiles arrive in row-major order within each chunk → sequential.
                    uint32_t tile_id = slab_start_tile;
                    for (uint32_t i = 0; i < Ht * Wt; ++i) {
                        output_cb.wait_front(1);
                        noc.async_write(output_cb, dst_accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
                        noc.async_write_barrier();
                        output_cb.pop_front(1);
                        tile_id++;
                    }
                } else {
                    // dim=-2: chunks along Wt. Each chunk: Ht rows × BLOCK_SIZE cols.
                    // Tiles arrive: (0, chunk*BS+0), (0, chunk*BS+1), ..., (1, chunk*BS+0), ...
                    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                        for (uint32_t ht = 0; ht < Ht; ++ht) {
                            for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
                                uint32_t wt = chunk * BLOCK_SIZE + i;
                                uint32_t tile_id = slab_start_tile + ht * Wt + wt;
                                output_cb.wait_front(1);
                                noc.async_write(
                                    output_cb, dst_accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
                                noc.async_write_barrier();
                                output_cb.pop_front(1);
                            }
                        }
                    }
                }
            } else if constexpr (dim == -1) {
                // chunk_along_reduce dim=-1: tiles arrive in row-major order
                uint32_t tile_id = slab_start_tile;
                for (uint32_t i = 0; i < Ht * Wt; ++i) {
                    output_cb.wait_front(1);
                    noc.async_write(output_cb, dst_accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
                    noc.async_write_barrier();
                    output_cb.pop_front(1);
                    tile_id++;
                }
            } else {
                // dim=-2: tiles arrive in column-major order
                // For wt=0: chunk 0 rows 0..BS-1, chunk 1 rows BS..2BS-1, ...
                // For wt=1: same pattern
                // tile_id = slab_start + ht * Wt + wt
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                        for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
                            uint32_t ht = chunk * BLOCK_SIZE + i;
                            uint32_t tile_id = slab_start_tile + ht * Wt + wt;
                            output_cb.wait_front(1);
                            noc.async_write(
                                output_cb, dst_accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
                            noc.async_write_barrier();
                            output_cb.pop_front(1);
                        }
                    }
                }
            }
            slab_start_tile += Ht * Wt;
        }
    } else {
        // ===== ROW_MAJOR path: write sticks from cb_rm_out =====
        //
        // write_sticks_after_untilize<cb_rm_out> waits/pops `width_in_tiles`
        // pages per block of 32 rows. The compute kernel's untilize helper
        // produces the same count. The writer just calls the helper.
        //
        // For chunk_along_reduce (dim=-1): each chunk writes 32 sticks
        //   (1 tile-row), BLOCK_SIZE tiles wide, using byte_offset_within_page
        //   to select the W-slice. 3 passes per tile-row.
        //
        // For chunk_along_reduce (dim=-2): each chunk writes BLOCK_SIZE*32
        //   sticks, 1 tile column wide, using byte_offset_within_page.
        //   3 passes per tile-column.
        //
        // For chunk_along_non_reduce (dim=-1): each chunk writes BLOCK_SIZE*32
        //   sticks, full W width. 1 pass per chunk.
        //
        // For chunk_along_non_reduce (dim=-2): each chunk writes origin_H sticks,
        //   BLOCK_SIZE tile columns wide. 1 pass per chunk.
        constexpr uint32_t tile_h = 32;
        const uint32_t tile_size = get_tile_size(cb_rm_out);
        const uint32_t full_row_bytes = origin_W * tile_size / (tile_h * tile_h);

        uint32_t slab_start_stick = start_id;

        for (uint32_t slab = 0; slab < num_slabs; ++slab) {
            if constexpr (chunk_along_reduce) {
                if constexpr (dim == -1) {
                    // Each chunk writes 32 sticks (1 tile-row), BLOCK_SIZE tiles wide
                    constexpr uint32_t chunk_row_bytes = BLOCK_SIZE * tile_size / tile_h;

                    for (uint32_t ht = 0; ht < Ht; ++ht) {
                        uint32_t base_stick = slab_start_stick + ht * tile_h;

                        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                            uint32_t byte_offset = chunk * chunk_row_bytes;
                            dataflow_kernel_lib::write_sticks_after_untilize<cb_rm_out>(
                                dst_accessor,
                                tile_h,           // total_num_rows (one tile-height of sticks)
                                chunk_row_bytes,  // row_bytes for this chunk
                                base_stick,       // start_page (stick index)
                                byte_offset       // byte_offset_within_page
                            );
                        }
                    }
                } else {
                    // dim=-2: each chunk writes BLOCK_SIZE*32 sticks, 1 tile column wide
                    constexpr uint32_t chunk_row_bytes = tile_size / tile_h;  // 1 tile column

                    for (uint32_t wt = 0; wt < Wt; ++wt) {
                        uint32_t byte_offset = wt * chunk_row_bytes;

                        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                            uint32_t base_stick = slab_start_stick + chunk * tile_h * BLOCK_SIZE;
                            dataflow_kernel_lib::write_sticks_after_untilize<cb_rm_out>(
                                dst_accessor,
                                tile_h * BLOCK_SIZE,  // total_num_rows
                                chunk_row_bytes,      // row_bytes (1 tile column)
                                base_stick,           // start_page
                                byte_offset           // byte_offset_within_page
                            );
                        }
                    }
                }
            } else {
                // chunk_along_non_reduce: 1 pass, full reduce dim per chunk
                if constexpr (dim == -1) {
                    // Each chunk: BLOCK_SIZE tile-rows × full W width
                    constexpr uint32_t chunk_row_bytes = BLOCK_SIZE * tile_size / tile_h;

                    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                        uint32_t base_stick = slab_start_stick + chunk * tile_h * BLOCK_SIZE;
                        dataflow_kernel_lib::write_sticks_after_untilize<cb_rm_out>(
                            dst_accessor,
                            tile_h * BLOCK_SIZE,  // total_num_rows (BLOCK_SIZE tile-rows)
                            full_row_bytes,       // row_bytes (full width)
                            base_stick,           // start_page
                            0                     // byte_offset_within_page
                        );
                    }
                } else {
                    // dim=-2: each chunk writes full H, BLOCK_SIZE tile-columns wide
                    constexpr uint32_t chunk_row_bytes = BLOCK_SIZE * tile_size / tile_h;

                    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                        uint32_t byte_offset = chunk * chunk_row_bytes;
                        dataflow_kernel_lib::write_sticks_after_untilize<cb_rm_out>(
                            dst_accessor,
                            origin_H,          // total_num_rows (full H)
                            chunk_row_bytes,   // row_bytes (BLOCK_SIZE tile columns)
                            slab_start_stick,  // start_page
                            byte_offset        // byte_offset_within_page
                        );
                    }
                }
            }
            slab_start_stick += origin_H;
        }
    }
}
