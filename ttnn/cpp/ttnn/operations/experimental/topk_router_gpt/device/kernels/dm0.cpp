// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DM0 Kernel: Weight + Input Reader (RISCV_0, NOC 0)
//
// Double-buffered block reads: push tiles in blocks of BLOCK_SIZE so
// compute can start matmul before all tiles arrive from DRAM.
//
// Optional: when hidden_rm_addr != 0 and group_id == 0, re-reads input
// tiles after the main loop and writes them untilized (RM) to DRAM.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t weight_addr = get_arg_val<uint32_t>(0);
    uint32_t input_addr = get_arg_val<uint32_t>(1);
    uint32_t dram_bank_id = get_arg_val<uint32_t>(2);
    uint32_t vchannel = get_arg_val<uint32_t>(3);
    uint32_t num_k_tiles = get_arg_val<uint32_t>(4);
    uint32_t k_tile_offset = get_arg_val<uint32_t>(5);
    uint32_t n_tile_id = get_arg_val<uint32_t>(6);
    uint32_t tile_size = get_arg_val<uint32_t>(7);
    uint32_t n_tiles_total = get_arg_val<uint32_t>(8);
    uint32_t is_worker = get_arg_val<uint32_t>(9);
    uint32_t bias_addr = get_arg_val<uint32_t>(10);
    uint32_t hidden_rm_addr = get_arg_val<uint32_t>(11);
    uint32_t hidden_rm_page_size = get_arg_val<uint32_t>(12);
    uint32_t group_id = get_arg_val<uint32_t>(13);

    constexpr uint32_t CB_WEIGHT = tt::CBIndex::c_0;
    constexpr uint32_t CB_INPUT = tt::CBIndex::c_1;
    constexpr uint32_t CB_BIAS = tt::CBIndex::c_4;

    const InterleavedAddrGenFast</*DRAM=*/true> input_addrgen = {
        .bank_base_address = input_addr, .page_size = tile_size, .data_format = get_dataformat(CB_INPUT)};

    const InterleavedAddrGenFast</*DRAM=*/true> weight_addrgen = {
        .bank_base_address = weight_addr, .page_size = tile_size, .data_format = get_dataformat(CB_WEIGHT)};

    // ----- Double-buffered block reads -----
    // Push tiles in blocks so compute can start matmul early.
    constexpr uint32_t BLOCK_SIZE = 2;
    uint32_t tiles_done = 0;

    while (tiles_done < num_k_tiles) {
        uint32_t block = num_k_tiles - tiles_done;
        if (block > BLOCK_SIZE) {
            block = BLOCK_SIZE;
        }

        cb_reserve_back(CB_INPUT, block);
        cb_reserve_back(CB_WEIGHT, block);
        uint32_t inp_wr = get_write_ptr(CB_INPUT);
        uint32_t wt_wr = get_write_ptr(CB_WEIGHT);

        for (uint32_t k = 0; k < block; k++) {
            uint32_t kg = k_tile_offset + tiles_done + k;
            noc_async_read_tile(kg, input_addrgen, inp_wr + k * tile_size);
            noc_async_read_tile(kg * n_tiles_total + n_tile_id, weight_addrgen, wt_wr + k * tile_size);
        }
        noc_async_read_barrier();
        cb_push_back(CB_INPUT, block);
        cb_push_back(CB_WEIGHT, block);

        tiles_done += block;
    }

    // ----- Read bias (worker only, 1 tile) -----
    if (is_worker) {
        const InterleavedAddrGenFast</*DRAM=*/true> bias_addrgen = {
            .bank_base_address = bias_addr, .page_size = tile_size, .data_format = get_dataformat(CB_BIAS)};

        cb_reserve_back(CB_BIAS, 1);
        uint32_t bias_write_ptr = get_write_ptr(CB_BIAS);
        noc_async_read_tile(n_tile_id, bias_addrgen, bias_write_ptr);
        noc_async_read_barrier();
        cb_push_back(CB_BIAS, 1);
    }

    // ----- Optional: untilize input to RM output (group 0 only) -----
    // Re-reads input tiles from DRAM, untilizes face→row layout, and writes
    // partial rows to a pre-allocated [B, hidden_dim] RM buffer in DRAM.
    // This overlaps with compute + DM1 during DM0's idle time.
    if (hidden_rm_addr != 0 && group_id == 0) {
        constexpr uint32_t CB_RM_SCRATCH = tt::CBIndex::c_18;

        const InterleavedAddrGen</*DRAM=*/true> rm_out_addrgen = {
            .bank_base_address = hidden_rm_addr, .page_size = hidden_rm_page_size};

        // Byte offset in each RM page for this core's column range
        uint32_t col_byte_offset = k_tile_offset * 32 * 2;

        for (uint32_t t = 0; t < num_k_tiles; t++) {
            uint32_t kg = k_tile_offset + t;

            // Re-read one input tile from DRAM into scratch
            cb_reserve_back(CB_RM_SCRATCH, 1);
            uint32_t scratch = get_write_ptr(CB_RM_SCRATCH);
            noc_async_read_tile(kg, input_addrgen, scratch);
            noc_async_read_barrier();

            // Software untilize: tile face layout → row-major
            // Tile layout: 4 faces of 16×16, face order: [top-left, top-right, bottom-left, bottom-right]
            // RM output: 32 rows × 32 cols, stored as 64 bytes per row
            volatile tt_l1_ptr uint32_t* tile32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
            // Use space after the tile for RM row buffer (32 bf16 = 64 bytes = 16 uint32)
            uint32_t rm_row_buf = scratch + tile_size;
            volatile tt_l1_ptr uint32_t* dst32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rm_row_buf);

            uint32_t tile_col_offset = t * 32 * 2;  // byte offset within this core's column range

            for (uint32_t row = 0; row < 32; row++) {
                // Extract one row from tile face layout: left half (cols 0-15) + right half (cols 16-31)
                uint32_t face_row = row & 15;
                uint32_t face_left = (row >= 16) ? 2 : 0;  // face 0 or 2
                uint32_t face_right = face_left + 1;       // face 1 or 3

                volatile tt_l1_ptr uint32_t* src_left = tile32 + face_left * 128 + face_row * 8;
                volatile tt_l1_ptr uint32_t* src_right = tile32 + face_right * 128 + face_row * 8;

                // Copy left half (8 uint32 = 16 bf16 = cols 0-15)
                for (uint32_t w = 0; w < 8; w++) {
                    dst32[w] = src_left[w];
                }
                // Copy right half (8 uint32 = 16 bf16 = cols 16-31)
                for (uint32_t w = 0; w < 8; w++) {
                    dst32[8 + w] = src_right[w];
                }

                // Write 64 bytes (one row of this tile) to the RM output
                uint64_t noc_addr = get_noc_addr(row, rm_out_addrgen) + col_byte_offset + tile_col_offset;
                noc_async_write(rm_row_buf, noc_addr, 64);
            }

            noc_async_write_barrier();
            cb_pop_front(CB_RM_SCRATCH, 1);
        }
    }
}
