// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax writer kernel (BRISC/RISCV_0).
//
// Per slab (one (N,C) pair):
//   TILE path:
//     - Reads Ht×Wt output tiles from cb_output_tiles and writes to DRAM/L1
//     - Tiles are in row-major tile order within each slab
//
//   ROW_MAJOR path:
//     - Reads Ht×Wt output sticks from cb_rm_out and writes to DRAM/L1
//       via write_sticks_after_untilize (always TILE granularity)
//     - Compute kernel untilizes cb_output_tiles → cb_rm_out
//
// CT args: Ht, Wt, is_rm, then TensorAccessorArgs starting at index 3
// RT args: output_buffer_address, start_id, num_slabs
//   TILE: start_id = starting tile index
//   RM:   start_id = starting stick (page) index

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
// CB indices — must match program descriptor
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

    // CT args: 3 scalar, then TensorAccessorArgs
    constexpr auto dst_args = TensorAccessorArgs<3>();
    const auto dst_accessor = TensorAccessor(dst_args, output_buffer_address);

    if constexpr (is_rm) {
        // ===== ROW_MAJOR path: write sticks from cb_rm_out =====
        // write_sticks_after_untilize reads Wt tile-sized pages from cb_rm_out
        // per tile-row, writes H sticks per call.
        // For tile-aligned shapes: row_bytes = W * elem_size = Wt * tile_size / 32.
        constexpr uint32_t tile_h = 32;
        const uint32_t tile_size = get_tile_size(cb_rm_out);
        const uint32_t row_bytes = Wt * tile_size / 32;

        // start_id is a stick (page) index in the RM tensor.
        // Each slab has H = Ht * 32 sticks.
        uint32_t stick_id = start_id;

        for (uint32_t slab = 0; slab < num_slabs; ++slab) {
            dataflow_kernel_lib::write_sticks_after_untilize<cb_rm_out>(
                dst_accessor,
                Ht * tile_h,  // total_num_rows = H (all rows at once for the slab)
                row_bytes,    // bytes per stick
                stick_id,     // start_page (stick index)
                0             // byte_offset_within_page
            );
            stick_id += Ht * tile_h;  // H sticks per slab
        }
    } else {
        // ===== TILE path: write tiles from cb_output_tiles =====
        CircularBuffer output_cb(cb_output_tiles);
        Noc noc;
        const uint32_t tile_bytes = get_tile_size(cb_output_tiles);

        uint32_t tiles_per_slab = Ht * Wt;
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
    }
}
