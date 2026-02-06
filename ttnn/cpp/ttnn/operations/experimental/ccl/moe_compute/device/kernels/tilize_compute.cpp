// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/tilize.h"

// Print a subset of a row-major bfloat16 buffer.
// BufferWidth: total number of columns (elements) per row
// cb_id: circular buffer ID
// cb_addr: base address of the CB data (caller must provide - use get_read_ptr or fifo_rd_ptr)
// start_row, end_row: row range to print [start_row, end_row)
// start_col, end_col: column range to print [start_col, end_col)
template <uint32_t BufferWidth>
void print_row_major_subset(
    uint32_t cb_addr, uint32_t start_row = 0, uint32_t end_row = 32, uint32_t start_col = 0, uint32_t end_col = 32) {
    volatile tt_l1_ptr uint16_t* data = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_addr);

    DPRINT << "=== Row-major [" << start_row << ":" << end_row << ", " << start_col << ":" << end_col
           << "] ===" << ENDL();

    for (uint32_t r = start_row; r < end_row; r++) {
        DPRINT << r << ": ";
        for (uint32_t c = start_col; c < end_col; c++) {
            uint32_t idx = r * BufferWidth + c;
            uint16_t val = data[idx];
            DPRINT << BF16(val);
            if (c < end_col - 1) {
                DPRINT << " ";
            }
        }
        DPRINT << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

void print_tile_rows(
    uint32_t cb_idx,
    uint32_t tile_idx,
    bool untilize = false,
    uint16_t start_row = 0,
    uint16_t end_row = 32,
    uint8_t start_col = 0,
    uint8_t end_col = 32) {
    DPRINT << "cb_idx: " << cb_idx << " tile_idx: " << tile_idx << ENDL();
    DPRINT << "======" << ENDL();
    for (uint16_t r = start_row; r < end_row; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_idx,
                      tile_idx,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)start_col,
                          .w1 = (uint8_t)end_col,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

// Compute kernel for tilizing incoming tokens from the reader.
// Waits for writer to pass total_chunks via CB, then processes each chunk:
// - Wait for reader to push tokens_per_chunk tokens
// - Tilize the block
// - Push tilized output to writer
namespace NAMESPACE {
void MAIN {
    // Compile-time arguments
    constexpr uint32_t tilize_input_cb_id = get_named_compile_time_arg_val("tilize_input_cb_id");
    constexpr uint32_t tilize_output_cb_id = get_named_compile_time_arg_val("tilize_output_cb_id");
    constexpr uint32_t total_chunks_cb_id = get_named_compile_time_arg_val("total_chunks_cb_id");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");
    constexpr uint32_t max_tiles_per_chunk = get_named_compile_time_arg_val("max_tiles_per_chunk");

    // Runtime arguments
    uint32_t rt_args_idx = 0;
    uint32_t tiles_per_chunk = get_arg_val<uint32_t>(rt_args_idx++);

    // Constants
    constexpr uint32_t one_page = 1;

    // Setup
    compute_kernel_hw_startup(tilize_input_cb_id, tilize_output_cb_id);
    fast_tilize_init(tilize_input_cb_id, max_tiles_per_chunk, tilize_output_cb_id);

    // Wait for writer to push total_chunks via CB
    cb_wait_front(total_chunks_cb_id, one_page);

    // Read total_chunks from the CB
    uint32_t total_chunks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(total_chunks_cb_id, 0));

    // Process each chunk
    for (uint32_t chunk = 0; chunk < total_chunks; chunk++) {
        // Wait for reader to push tokens_per_chunk pages (row-major data)
        // Reader always reserves/pushes tokens_per_chunk for consistent synchronization
        cb_wait_front(tilize_input_cb_id, tokens_per_chunk);

        // DEBUG: Print subsets of input (row-major bfloat16)
        // Get CB address directly within UNPACK context (avoids mailbox sync issues with get_tile_address)
        // UNPACK(({
        //     uint32_t operand_id = get_operand_id(tilize_input_cb_id);
        //     uint32_t cb_addr = get_local_cb_interface(operand_id).fifo_rd_ptr << 4;  // Convert to byte address
        //     DPRINT << "=== CHUNK " << chunk << " INPUT ===" << ENDL();
        //     print_row_major_subset<buffer_width>(cb_addr, 0, 1, 0, 1);  // first 4 rows x 8 cols
        //     print_row_major_subset<buffer_width>(cb_addr, 0, 4, buffer_width - 8, buffer_width);  // last 4x8
        // }));

        cb_reserve_back(tilize_output_cb_id, tiles_per_chunk);

        fast_tilize_block(tilize_input_cb_id, tiles_per_chunk, tilize_output_cb_id);

        // DEBUG: Print first and last tiles of output (tilized format)
        // PACK(({
        //     DPRINT << "=== CHUNK " << chunk << " OUTPUT ===" << ENDL();
        //     print_tile_rows(tilize_output_cb_id, 0, true, 0, 1, 0, 1);                    // First tile, 4x8
        //     // print_tile_rows(tilize_output_cb_id, tiles_per_chunk - 1, true, 0, 1, 0, 1);  // Last tile, 4x8
        // }));

        cb_push_back(tilize_output_cb_id, tiles_per_chunk);
    }

    fast_tilize_uninit(tilize_input_cb_id, tilize_output_cb_id);
    cb_pop_front(total_chunks_cb_id, one_page);
}
}  // namespace NAMESPACE
