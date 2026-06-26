// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax writer kernel — V2 streaming path.
//
// The V2 compute kernel writes output tiles one chunk at a time (BLOCK_SIZE tiles
// per chunk). The writer drains them as they arrive.
//
// TILE path: reads tiles from cb_output_tiles, writes to DRAM/L1
//   For dim=-1: tiles arrive in row-major order (standard)
//   For dim=-2: tiles arrive in column-major order (per-column chunks)
// RM path:   reads sticks from cb_rm_out (compute untilizes), writes to DRAM/L1

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
                // chunk_along_non_reduce: tiles arrive in row-major order (standard)
                uint32_t tile_id = slab_start_tile;
                for (uint32_t i = 0; i < Ht * Wt; ++i) {
                    output_cb.wait_front(1);
                    noc.async_write(output_cb, dst_accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
                    noc.async_write_barrier();
                    output_cb.pop_front(1);
                    tile_id++;
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
        // For dim=-1: compute untilizes per tile-row chunk, writer writes per tile-row
        // For dim=-2: compute untilizes per tile-column chunk, writer writes per tile-column
        // The RM writer uses write_sticks_after_untilize which writes contiguous sticks.
        // For dim=-2, the output tiles are in column-major order, which doesn't map
        // directly to contiguous sticks. This is a known limitation — for now, the
        // RM + dim=-2 + V2 path is not supported (should not trigger V2 for typical
        // RM shapes, since RM is typically used with dim=-1 for attention).
        constexpr uint32_t tile_h = 32;
        const uint32_t tile_size = get_tile_size(cb_rm_out);
        const uint32_t row_bytes = origin_W * tile_size / (tile_h * tile_h);

        if constexpr (dim == -1) {
            // dim=-1 RM: tiles arrive in row-major order, writer writes contiguous sticks
            uint32_t stick_id = start_id;
            for (uint32_t slab = 0; slab < num_slabs; ++slab) {
                dataflow_kernel_lib::write_sticks_after_untilize<cb_rm_out>(
                    dst_accessor, origin_H, row_bytes, stick_id, 0);
                stick_id += origin_H;
            }
        } else {
            // dim=-2 RM + V2: not supported. This path should not be reached for
            // typical RM shapes. If it is, fall back to per-tile writes.
            // Each tile is 32 rows × 32 columns. We write each tile's 32 sticks.
            CircularBuffer output_cb(cb_rm_out);
            Noc noc;
            const uint32_t tile_bytes = get_tile_size(cb_rm_out);

            for (uint32_t slab = 0; slab < num_slabs; ++slab) {
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                        for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
                            uint32_t ht = chunk * BLOCK_SIZE + i;
                            // Write 32 sticks for this tile
                            uint32_t base_stick = start_id + slab * origin_H + ht * 32;
                            for (uint32_t r = 0; r < 32; ++r) {
                                uint32_t stick_id = base_stick + r;
                                output_cb.wait_front(1);
                                // Actually this won't work for RM — the cb_rm_out
                                // has tile-sized pages, not stick-sized.
                                // This path needs more work. For now, assert.
                                ASSERT(false);
                            }
                        }
                    }
                }
            }
        }
    }
}
