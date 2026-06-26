// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax writer kernel — V2 streaming path.
//
// The V2 compute kernel writes output tiles one chunk at a time (BLOCK_SIZE tiles
// per chunk). The writer drains them as they arrive.
//
// TILE path: reads tiles from cb_output_tiles, writes to DRAM/L1
// RM path:   reads sticks from cb_rm_out (compute untilizes), writes to DRAM/L1
//
// The writer receives one tile at a time (streaming) for TILE, or one
// tile-height block at a time for RM.

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

    constexpr auto dst_args = TensorAccessorArgs<5>();
    const auto dst_accessor = TensorAccessor(dst_args, output_buffer_address);

    constexpr uint32_t tiles_per_slab = Ht * Wt;

    if constexpr (!is_rm) {
        // ===== TILE path: write tiles from cb_output_tiles =====
        CircularBuffer output_cb(cb_output_tiles);
        Noc noc;
        const uint32_t tile_bytes = get_tile_size(cb_output_tiles);

        uint32_t tile_id = start_id;
        for (uint32_t slab = 0; slab < num_slabs; ++slab) {
            for (uint32_t i = 0; i < tiles_per_slab; ++i) {
                output_cb.wait_front(1);
                noc.async_write(output_cb, dst_accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
                noc.async_write_barrier();
                output_cb.pop_front(1);
                tile_id++;
            }
        }
    } else {
        // ===== ROW_MAJOR path: write sticks from cb_rm_out =====
        // Compute untilizes in chunks of BLOCK_SIZE tiles per tile-row.
        // write_sticks_after_untilize reads tile-sized pages from cb_rm_out.
        constexpr uint32_t tile_h = 32;
        const uint32_t tile_size = get_tile_size(cb_rm_out);
        const uint32_t row_bytes = origin_W * tile_size / (tile_h * tile_h);

        uint32_t stick_id = start_id;
        for (uint32_t slab = 0; slab < num_slabs; ++slab) {
            dataflow_kernel_lib::write_sticks_after_untilize<cb_rm_out>(
                dst_accessor,
                origin_H,   // total_num_rows = actual H
                row_bytes,  // bytes per stick
                stick_id,   // start_page
                0           // byte_offset_within_page
            );
            stick_id += origin_H;
        }
    }
}
