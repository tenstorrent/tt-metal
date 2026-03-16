// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Utility helpers for layernorm dataflow kernels with RM output.

#pragma once

#include <stdint.h>

namespace norm::layernorm::device::kernels::dataflow {

/**
 * @brief Write one untilized block of row-major data from a circular buffer to DRAM.
 *
 * The compute kernel's pack_untilize_block produces a (TILE_H x block_size*TILE_W)
 * row-major array in the CB.  This helper drains that array by writing one full-width
 * stick per valid row via the NOC.
 *
 * @tparam TensorAccessorT  Type of the tensor accessor (deduced from dst_a).
 * @tparam BlockT           Type of the block descriptor (deduced from block).
 * @tparam TILE_W           Tile width in elements (typically 32).
 * @tparam TILE_H           Tile height in rows (typically 32).
 *
 * @param cb_id               Circular buffer index to read from (e.g., c_28).
 * @param dst_a               TensorAccessor for the DRAM output buffer.
 * @param abs_row_base        Absolute row index of the first row in this tile-row.
 * @param num_valid_rows      Number of valid (non-padding) rows to write (0..TILE_H).
 * @param tile_width_bytes    TILE_W * elem_size_bytes.
 * @param block_row_stride_bytes  block_size * TILE_W * elem_size_bytes (row stride in CB).
 * @param block               Block descriptor from BlockedRange iteration.
 */
template <typename TensorAccessorT, typename BlockT, uint32_t TILE_W, uint32_t TILE_H>
inline void write_row_major_block_from_cb(
    uint32_t cb_id,
    TensorAccessorT& dst_a,
    uint32_t abs_row_base,
    uint32_t num_valid_rows,
    uint32_t tile_width_bytes,
    uint32_t block_row_stride_bytes,
    BlockT block) {
    // Wait for untilized block data in the CB
    cb_wait_front(cb_id, block.size());

    uint32_t l1_read_addr = get_read_ptr(cb_id);

    // Write each valid row to DRAM.
    // The block holds block.size() tiles worth of row-major data.
    // Row r starts at: l1_read_addr + r * block_row_stride_bytes + block_local_col_offset.
    // block_local_col_offset = 0 because the block always starts at tile column
    // block.start(), but the data in the CB is laid out contiguously from the block start.
    //
    // The NOC write for each row covers block.size() * tile_width_bytes of data.
    uint32_t write_size_bytes = block.size() * tile_width_bytes;

    for (uint32_t row = 0; row < num_valid_rows; row++) {
        uint32_t l1_addr = l1_read_addr + row * block_row_stride_bytes;
        uint32_t dram_row = abs_row_base + row;
        // page_id = dram_row (one page per full-width row in the output tensor).
        // The block writes a portion of the row starting at column block.start() * TILE_W.
        uint32_t col_offset_bytes = block.start() * tile_width_bytes;
        uint64_t noc_addr = dst_a.get_noc_addr(dram_row, col_offset_bytes);
        noc_async_write(l1_addr, noc_addr, write_size_bytes);
    }

    noc_async_write_barrier();
    cb_pop_front(cb_id, block.size());
}

}  // namespace norm::layernorm::device::kernels::dataflow
