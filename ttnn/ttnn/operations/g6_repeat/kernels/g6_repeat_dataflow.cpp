// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_repeat: out tile (t, w) = x tile w with row 0 copied into rows 0..rows_to_fill-1.
//
// One data-movement kernel per active core owns cb_repeat_tiles as workspace: no
// other thread touches it, so there is no CB handshake, only NoC barriers.
// Per block of out tile columns:
//   load_row0_faces_block  x upper faces (row 0 + padding rows) land in the out slots
//   replicate_row0_block   row 0 -> rows 1..rows_to_fill-1 by self-aimed NoC reads
//   store_block            each slot written once to its out page
// then write_barrier: every write acked before the kernel returns.
//
// Raw-API entries (load, store, barriers) have no kernel_lib helper: kernel_lib has no
// TensorAccessor tile-page streaming helper, and noc_async_read_page cannot read a
// partial page. The replicate uses dataflow_kernel_lib's stateful local-copy helpers.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_repeat_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t repeat_buffer_depth = get_compile_time_arg_val(2);
    constexpr uint32_t rows_to_fill = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_out_wt_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_out_ht_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t out_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t x_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t row0_faces_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t face_cols = get_compile_time_arg_val(11);
    constexpr uint32_t face_h = get_compile_time_arg_val(12);
    constexpr auto x_args = TensorAccessorArgs<13>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    constexpr uint32_t slot_set_bytes = block_width_tiles * out_page_size;

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t start_tile_col = get_arg_val<uint32_t>(0);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(1);

    const auto x_accessor = TensorAccessor(x_args, x_addr, x_page_size);
    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page_size);

    const uint32_t slots_base = get_write_ptr(cb_repeat_tiles);
    const uint32_t num_blocks = (core_w_tiles + block_width_tiles - 1) / block_width_tiles;

    Noc noc;

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_start_col = start_tile_col + block_idx * block_width_tiles;
        const uint32_t cols_left = core_w_tiles - block_idx * block_width_tiles;
        const uint32_t block_extent = cols_left < block_width_tiles ? cols_left : block_width_tiles;
        const uint32_t slot_set_base = slots_base + (block_idx % repeat_buffer_depth) * slot_set_bytes;

        {
            DeviceZoneScopedN("load_row0_faces_block");
            // The NoC reads a write source after issue: the writes that last used this
            // slot set must have left L1 before new x data lands on it.
            if (block_idx >= repeat_buffer_depth) {
                noc_async_writes_flushed();
            }
            // Read starts at page offset 0 and lands at a slot base: both ends stay
            // DRAM-read aligned.
            uint32_t slot = slot_set_base;
            for (uint32_t j = 0; j < block_extent; ++j) {
                noc_async_read(x_accessor.get_noc_addr(block_start_col + j, 0), slot, row0_faces_bytes);
                slot += out_page_size;
            }
            noc_async_read_barrier();
        }

        if constexpr (rows_to_fill > 1) {
            DeviceZoneScopedN("replicate_row0_block");
            // Must follow the load barrier: the load also writes padding into rows 1..face_h-1,
            // which would overwrite an earlier copy.
            uint32_t slot = slot_set_base;
            for (uint32_t j = 0; j < block_extent; ++j) {
                for (uint32_t c = 0; c < face_cols; ++c) {
                    const uint32_t src = slot + c * face_bytes;
                    dataflow_kernel_lib::set_read_state<face_row_bytes>(noc, src);
                    for (uint32_t r = 1; r < rows_to_fill; ++r) {
                        // One row of faces spans row0_faces_bytes, so that is the stride to faces 2/3.
                        const uint32_t dst = src + (r / face_h) * row0_faces_bytes + (r % face_h) * face_row_bytes;
                        dataflow_kernel_lib::read_with_state<face_row_bytes>(noc, dst, src);
                    }
                }
                slot += out_page_size;
            }
            noc_async_read_barrier();
        }

        {
            DeviceZoneScopedN("store_block");
            uint32_t slot = slot_set_base;
            for (uint32_t j = 0; j < block_extent; ++j) {
                const uint32_t col = block_start_col + j;
                for (uint32_t t = 0; t < tensor_out_ht_tiles; ++t) {
                    noc_async_write_page(t * tensor_out_wt_tiles + col, out_accessor, slot);
                }
                slot += out_page_size;
            }
        }
    }

    {
        // The next program starts right after this one: no write may stay in flight.
        DeviceZoneScopedN("write_barrier");
        noc_async_write_barrier();
    }
}
