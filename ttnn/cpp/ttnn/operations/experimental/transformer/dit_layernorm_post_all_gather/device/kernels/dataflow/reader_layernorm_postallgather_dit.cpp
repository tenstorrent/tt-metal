// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Reads LayerNorm inputs, gathered stats, optional gamma/beta, and epsilon from interleaved DRAM.
 * LayerNorm-only; Welford-only; non-sharded; no 2D variants.
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/tt-metalium/constants.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/debug/assert.h"
#include "api/debug/dprint.h"

// Reads a single row from DRAM and broadcasts it to all rows in a tile.
// For TILE layout: read row 0 from face 0&1, copy to face 2&3
// For ROW_MAJOR layout: read row from DRAM, copy within L1 to face 2&3
template <uint32_t is_row_major, uint32_t element_size>
void async_read_row_to_tile(const uint64_t DRAM_src_addr, uint32_t L1_dst_addr);

void kernel_main() {
    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_stats = tt::CBIndex::c_1;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta = tt::CBIndex::c_3;
    constexpr uint32_t cb_eps = tt::CBIndex::c_4;

    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);
    const uint32_t stats_tile_bytes = get_tile_size(cb_stats);

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(1);
    constexpr uint32_t gamma_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t beta_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t gamma_is_row_major = get_compile_time_arg_val(4);
    constexpr uint32_t beta_is_row_major = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t gamma_element_size = get_compile_time_arg_val(7);
    constexpr uint32_t beta_element_size = get_compile_time_arg_val(8);
    constexpr uint32_t gamma_is_batched = get_compile_time_arg_val(9);
    constexpr uint32_t beta_is_batched = get_compile_time_arg_val(10);
    constexpr uint32_t gamma_batch_stride_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t beta_batch_stride_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t Ht = get_compile_time_arg_val(13);
    constexpr auto src_args = TensorAccessorArgs<14>();
    constexpr auto stats_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(arg_idx++);    // Source address in dram
    const uint32_t stats_addr = get_arg_val<uint32_t>(arg_idx++);  // Source address in dram
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t beta_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eps = get_arg_val<uint32_t>(arg_idx++);
    generate_bcast_col_scalar(cb_eps, eps);

    const auto src_a = TensorAccessor(src_args, src_addr, src0_tile_bytes);
    const auto src_stats = TensorAccessor(stats_args, stats_addr, stats_tile_bytes);

#ifdef FUSE_GAMMA
    const auto addrg = TensorAccessor(gamma_args, gamma_addr, gamma_page_size);
    const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
#endif
#ifdef FUSE_BETA
    const auto addrb = TensorAccessor(beta_args, beta_addr, beta_page_size);
    const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
#endif

    // Track the current batch for batched gamma/beta
    // batch_idx = tile_row / Ht (Ht = tile rows per sequence position in NC dimension)
    uint32_t current_gamma_batch = tile_row_start / Ht;
    uint32_t current_beta_batch = tile_row_start / Ht;
    bool gamma_loaded_for_batch = false;
    bool beta_loaded_for_batch = false;

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        uint32_t stats_tile_idx = tile_row * stats_tiles_cols;
        cb_reserve_back(cb_stats, stats_tiles_cols);
        DPRINT << "reserve_back stats on tile_row: " << tile_row << ENDL();
        uint32_t stats_wr_ptr = get_write_ptr(cb_stats);
        for (uint32_t st = 0; st < stats_tiles_cols; ++st) {
            noc_async_read_tile(stats_tile_idx, src_stats, stats_wr_ptr);
            stats_wr_ptr += stats_tile_bytes;
            stats_tile_idx++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_stats, stats_tiles_cols);

        // Check if we've switched to a new batch
        uint32_t batch_idx = tile_row / Ht;

#ifdef FUSE_GAMMA
        bool need_new_gamma = false;
        if constexpr (gamma_is_batched) {
            if (batch_idx != current_gamma_batch || !gamma_loaded_for_batch) {
                need_new_gamma = true;
                current_gamma_batch = batch_idx;
            }
        } else {
            // Non-batched: only load on first tile_row
            need_new_gamma = (tile_row == tile_row_start);
        }
#endif
#ifdef FUSE_BETA
        bool need_new_beta = false;
        if constexpr (beta_is_batched) {
            if (batch_idx != current_beta_batch || !beta_loaded_for_batch) {
                need_new_beta = true;
                current_beta_batch = batch_idx;
            }
        } else {
            // Non-batched: only load on first tile_row
            need_new_beta = (tile_row == tile_row_start);
        }
#endif

        // Loop tiles in blocks of block_size
        uint32_t input_tile_idx = tile_row * Wt;
        for (uint32_t col_tile = 0; col_tile < Wt; col_tile += block_size) {
            // Input
            cb_reserve_back(cb_inp, block_size);
            DPRINT << "reserve_back input on tile_row: " << tile_row << " col_tile: " << col_tile << ENDL();
            uint32_t inp_wr_ptr = get_write_ptr(cb_inp);
            for (uint32_t i = 0; i < block_size && col_tile + i < Wt; i++) {
                noc_async_read_tile(input_tile_idx, src_a, inp_wr_ptr);
                inp_wr_ptr += src0_tile_bytes;
                input_tile_idx++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_inp, block_size);

#ifdef FUSE_GAMMA
            if (need_new_gamma) {
                // Read in gamma block-by-block for this batch
                cb_reserve_back(cb_gamma, block_size);
                DPRINT << "reserve_back gamma on tile_row: " << tile_row << " col_tile: " << col_tile
                       << " batch: " << batch_idx << ENDL();
                uint32_t l1_write_addr_g = get_write_ptr(cb_gamma);
                // Calculate tile offset for this batch
                uint32_t gamma_batch_offset = gamma_is_batched ? (batch_idx * gamma_batch_stride_tiles) : 0;
                for (uint32_t i = 0; i < block_size && col_tile + i < Wt; i++) {
                    uint64_t gamma_noc_addr = get_noc_addr(gamma_batch_offset + col_tile + i, addrg);
                    async_read_row_to_tile<gamma_is_row_major, gamma_element_size>(gamma_noc_addr, l1_write_addr_g);
                    l1_write_addr_g += gamma_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_gamma, block_size);
            }
#endif
#ifdef FUSE_BETA
            if (need_new_beta) {
                // Read in beta block-by-block for this batch
                cb_reserve_back(cb_beta, block_size);
                DPRINT << "reserve_back beta on tile_row: " << tile_row << " col_tile: " << col_tile
                       << " batch: " << batch_idx << ENDL();
                uint32_t l1_write_addr_b = get_write_ptr(cb_beta);
                // Calculate tile offset for this batch
                uint32_t beta_batch_offset = beta_is_batched ? (batch_idx * beta_batch_stride_tiles) : 0;
                for (uint32_t i = 0; i < block_size && col_tile + i < Wt; i++) {
                    uint64_t beta_noc_addr = get_noc_addr(beta_batch_offset + col_tile + i, addrb);
                    async_read_row_to_tile<beta_is_row_major, beta_element_size>(beta_noc_addr, l1_write_addr_b);
                    l1_write_addr_b += beta_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_beta, block_size);
            }
#endif
        }

#ifdef FUSE_GAMMA
        if (need_new_gamma) {
            gamma_loaded_for_batch = true;
        }
#endif
#ifdef FUSE_BETA
        if (need_new_beta) {
            beta_loaded_for_batch = true;
        }
#endif
    }
}

// Reads a single row (32 elements) from DRAM and places it in tile format for row broadcast.
// Tile memory layout: Face 0 (256 elems), Face 1 (256 elems), Face 2 (256 elems), Face 3 (256 elems)
// For row broadcast, we read the first row and copy it to the position for faces 2&3.
template <uint32_t is_row_major, uint32_t element_size>
void async_read_row_to_tile(const uint64_t DRAM_src_addr, uint32_t L1_dst_addr) {
    // Byte sizes for tile layout
    constexpr uint32_t face_row_bytes = tt::constants::FACE_WIDTH * element_size;  // 16 elements per row
    constexpr uint32_t tile_row_bytes = tt::constants::TILE_WIDTH * element_size;  // 32 elements per row
    constexpr uint32_t single_face_bytes = tt::constants::FACE_HW * element_size;  // 16*16 = 256 elements

    // Read row 0 into face 0 (first 32 elements of the tile)
    noc_async_read(DRAM_src_addr, L1_dst_addr, tile_row_bytes);

    if constexpr (is_row_major == 0) {  // TILE layout
        // For TILE layout, face 1 row 0 is at offset single_face_bytes in DRAM, read it
        noc_async_read(DRAM_src_addr + single_face_bytes, L1_dst_addr + single_face_bytes, face_row_bytes);
    } else if constexpr (is_row_major == 1) {  // ROW_MAJOR layout
        // For ROW_MAJOR, source is 1D row, copy the data we read to face 1 position
        noc_async_read_barrier();
        uint64_t l1_noc_addr = get_noc_addr(L1_dst_addr + face_row_bytes);
        noc_async_read(l1_noc_addr, L1_dst_addr + single_face_bytes, face_row_bytes);
    } else {
        static_assert(is_row_major == 0 || is_row_major == 1, "Layout must be ROW_MAJOR (1) or TILE (0)");
    }
}
