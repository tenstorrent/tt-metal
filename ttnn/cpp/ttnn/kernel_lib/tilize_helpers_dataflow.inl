// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file tilize_helpers_dataflow.inl
 * @brief Implementation of tilize/untilize dataflow helpers
 *
 * This file contains the implementation details for read_sticks_for_tilize()
 * and write_sticks_after_untilize(). It should only be included by
 * tilize_helpers_dataflow.hpp.
 *
 * These helpers pair with the compute-side helpers in:
 *   - tilize_helpers.hpp  (compute_kernel_lib::tilize)
 *   - untilize_helpers.hpp (compute_kernel_lib::untilize)
 */

namespace dataflow_kernel_lib {

namespace detail {

constexpr uint32_t round_up(uint32_t value, uint32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

constexpr uint32_t div_up(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

}  // namespace detail

// ─── read_sticks_for_tilize ─────────────────────────────────────────────────
//
// TILE granularity example (reader kernel, single-core):
//
//   #include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
//   void kernel_main() {
//       constexpr auto src_args = TensorAccessorArgs<0>();
//       const auto accessor = TensorAccessor(src_args, get_arg_val<uint32_t>(0));
//       dataflow_kernel_lib::read_sticks_for_tilize<cb_in>(
//           accessor, total_num_rows, row_bytes);
//   }
//   // Compute side (symmetric tilize — tilize_helpers.hpp):
//   //   compute_kernel_lib::tilize<width_tiles, cb_in, cb_out>(num_blocks);
//   // CB config: page_size = tile_size
//
// Multi-core example (reader kernel):
//
//   void kernel_main() {
//       uint32_t start_row = get_arg_val<uint32_t>(1);  // per-core offset
//       uint32_t num_rows  = get_arg_val<uint32_t>(2);  // per-core count
//       dataflow_kernel_lib::read_sticks_for_tilize<cb_in>(
//           accessor, num_rows, row_bytes, start_row);
//   }
//
// ROW granularity example (reader kernel):
//
//   dataflow_kernel_lib::read_sticks_for_tilize<cb_in, TilizeGranularity::ROW>(
//       accessor, total_num_rows, row_bytes);
//   // Compute side (asymmetric tilize — tilize_helpers.hpp):
//   //   compute_kernel_lib::tilize<width_tiles, cb_in, cb_out>(num_blocks, total_num_rows);
//   // CB config: page_size = padded_row_bytes
//   // L1 benefit: when total_num_rows < 32, CB only needs total_num_rows pages
//   //   instead of width_in_tiles tile-pages (which always span 32 rows).
//
template <uint32_t cb_id, TilizeGranularity granularity, typename Accessor>
FORCE_INLINE void read_sticks_for_tilize(
    const Accessor& accessor, uint32_t total_num_rows, uint32_t row_bytes, uint32_t start_page) {
    // Derive tile geometry from CB configuration (all constexpr)
    constexpr uint32_t tile_h = unpack_tile_r_dim[cb_id];
    constexpr uint32_t tile_w = unpack_tile_c_dim[cb_id];
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    constexpr uint32_t tile_hw = get_tile_hw(cb_id);
    constexpr uint32_t elem_size = tile_size / tile_hw;
    constexpr uint32_t tile_row_bytes = tile_w * elem_size;

    // Block-float formats (BFP8, BFP4, etc.) have tile overhead that breaks
    // the elem_size = tile_size / tile_hw derivation. These formats are never
    // used with row-major data, so this helper should not be called for them.
    ASSERT(tile_size % tile_hw == 0);

    // Basic sanity
    ASSERT(total_num_rows > 0);
    ASSERT(row_bytes > 0);

    // Pad row width up to tile boundary
    uint32_t padded_row_bytes = detail::round_up(row_bytes, tile_row_bytes);
    uint32_t width_in_tiles = padded_row_bytes / tile_row_bytes;
    uint32_t total_blocks = detail::div_up(total_num_rows, tile_h);

    if constexpr (granularity == TilizeGranularity::TILE) {
        // ── TILE mode ───────────────────────────────────────────────────
        // Each block: reserve width_in_tiles pages, read tile_h rows, push.
        // CB page_size = tile_size.
        // Pairs with compute_kernel_lib::tilize(num_blocks) [symmetric].

        // Deadlock prevention: compute waits for width_in_tiles pages per block.
        // If CB capacity < width_in_tiles, cb_reserve_back blocks forever because
        // compute never pops (it hasn't received a full block yet).
        uint32_t cb_capacity = get_local_cb_interface(cb_id).fifo_num_pages;
        if (cb_capacity > 0) {
            ASSERT(width_in_tiles <= cb_capacity);
        }

        for (uint32_t block = 0; block < total_blocks; block++) {
            uint32_t block_row = block * tile_h;
            uint32_t rows_this_block = total_num_rows - block_row;
            if (rows_this_block > tile_h) {
                rows_this_block = tile_h;
            }

            cb_reserve_back(cb_id, width_in_tiles);
            uint32_t l1_addr = get_write_ptr(cb_id);

            for (uint32_t row = 0; row < rows_this_block; row++) {
                uint64_t noc_addr = accessor.get_noc_addr(start_page + block_row + row);
                noc_async_read(noc_addr, l1_addr, row_bytes);
                l1_addr += padded_row_bytes;
            }

            noc_async_read_barrier();
            cb_push_back(cb_id, width_in_tiles);
        }
    } else {
        // ── ROW mode ────────────────────────────────────────────────────
        // Each row: reserve 1 page, read 1 row, push.
        // CB page_size = padded_row_bytes.
        // Pairs with compute_kernel_lib::tilize(num_blocks, total_num_rows) [asymmetric].
        // The asymmetric tilize waits for min(32, pages_left) row-pages per block.
        // L1 advantage: CB only needs min(tile_h, total_num_rows) pages buffered,
        // which saves space when total_num_rows < tile_h (e.g. 7 rows needs
        // 7 * padded_row_bytes instead of width_in_tiles * tile_size).

        // Deadlock prevention: asymmetric tilize waits for min(tile_h, total_num_rows)
        // row-pages in the first block. If CB can't hold that many pages, the reader
        // blocks on cb_reserve_back while compute is stuck waiting for more rows.
        uint32_t rows_first_block = (total_num_rows < tile_h) ? total_num_rows : tile_h;
        uint32_t cb_capacity = get_local_cb_interface(cb_id).fifo_num_pages;
        if (cb_capacity > 0) {
            ASSERT(rows_first_block <= cb_capacity);
        }

        for (uint32_t row = 0; row < total_num_rows; row++) {
            cb_reserve_back(cb_id, 1);
            uint32_t l1_addr = get_write_ptr(cb_id);

            uint64_t noc_addr = accessor.get_noc_addr(start_page + row);
            noc_async_read(noc_addr, l1_addr, row_bytes);

            noc_async_read_barrier();
            cb_push_back(cb_id, 1);
        }
    }
}

// ─── write_sticks_after_untilize ────────────────────────────────────────────
//
// Example (writer kernel, single-core):
//
//   #include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
//   void kernel_main() {
//       constexpr auto dst_args = TensorAccessorArgs<0>();
//       const auto accessor = TensorAccessor(dst_args, get_arg_val<uint32_t>(0));
//       dataflow_kernel_lib::write_sticks_after_untilize<cb_out>(
//           accessor, total_num_rows, row_bytes);
//   }
//   // Compute side (untilize_helpers.hpp):
//   //   compute_kernel_lib::untilize<width_tiles, cb_in, cb_out>(num_blocks);
//   // CB config: page_size = tile_size (untilize always outputs tile-sized pages)
//
// Multi-core example (writer kernel):
//
//   void kernel_main() {
//       uint32_t start_row = get_arg_val<uint32_t>(1);  // per-core offset
//       uint32_t num_rows  = get_arg_val<uint32_t>(2);  // per-core count
//       dataflow_kernel_lib::write_sticks_after_untilize<cb_out>(
//           accessor, num_rows, row_bytes, start_row);
//   }
//
template <uint32_t cb_id, typename Accessor>
FORCE_INLINE void write_sticks_after_untilize(
    const Accessor& accessor, uint32_t total_num_rows, uint32_t row_bytes, uint32_t start_page) {
    // Derive tile geometry from CB configuration (all constexpr)
    constexpr uint32_t tile_h = unpack_tile_r_dim[cb_id];
    constexpr uint32_t tile_w = unpack_tile_c_dim[cb_id];
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    constexpr uint32_t tile_hw = get_tile_hw(cb_id);
    constexpr uint32_t elem_size = tile_size / tile_hw;
    constexpr uint32_t tile_row_bytes = tile_w * elem_size;

    // Block-float format guard (same as read helper)
    ASSERT(tile_size % tile_hw == 0);

    // Basic sanity
    ASSERT(total_num_rows > 0);
    ASSERT(row_bytes > 0);

    // Pad row width up to tile boundary
    uint32_t padded_row_bytes = detail::round_up(row_bytes, tile_row_bytes);
    uint32_t width_in_tiles = padded_row_bytes / tile_row_bytes;
    uint32_t total_blocks = detail::div_up(total_num_rows, tile_h);

    // Always TILE granularity — compute_kernel_lib::untilize produces tile-sized pages.
    // Pairs with compute_kernel_lib::untilize<width_tiles, cb_in, cb_out>(num_blocks).

    // Deadlock prevention: writer waits for width_in_tiles pages from compute.
    // If CB capacity < width_in_tiles, cb_wait_front blocks forever.
    uint32_t cb_capacity = get_local_cb_interface(cb_id).fifo_num_pages;
    if (cb_capacity > 0) {
        ASSERT(width_in_tiles <= cb_capacity);
    }

    for (uint32_t block = 0; block < total_blocks; block++) {
        uint32_t block_row = block * tile_h;
        uint32_t rows_this_block = total_num_rows - block_row;
        if (rows_this_block > tile_h) {
            rows_this_block = tile_h;
        }

        cb_wait_front(cb_id, width_in_tiles);
        uint32_t l1_addr = get_read_ptr(cb_id);

        for (uint32_t row = 0; row < rows_this_block; row++) {
            uint64_t noc_addr = accessor.get_noc_addr(start_page + block_row + row);
            noc_async_write(l1_addr, noc_addr, row_bytes);
            l1_addr += padded_row_bytes;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_id, width_in_tiles);
    }
}

}  // namespace dataflow_kernel_lib
