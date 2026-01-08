// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Reads LayerNorm inputs, gathered stats, optional gamma/beta, and epsilon from interleaved DRAM.
 * LayerNorm-only; Welford-only; non-sharded; no 2D variants.
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "api/debug/assert.h"
#include "api/debug/dprint.h"

template <uint32_t t>
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
    constexpr auto src_args = TensorAccessorArgs<7>();
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
            if (tile_row == tile_row_start) {
                // Read in gamma block-by-block when on the first row
                cb_reserve_back(cb_gamma, block_size);
                DPRINT << "reserve_back gamma on tile_row: " << tile_row << " col_tile: " << col_tile << ENDL();
                uint32_t l1_write_addr_g = get_write_ptr(cb_gamma);
                for (uint32_t i = 0; i < block_size && col_tile + i < Wt; i++) {
                    uint64_t gamma_noc_addr = get_noc_addr(col_tile + i, addrg);
                    async_read_row_to_tile<gamma_is_row_major>(gamma_noc_addr, l1_write_addr_g);
                    l1_write_addr_g += gamma_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_gamma, block_size);
            }
#endif
#ifdef FUSE_BETA
            if (tile_row == tile_row_start) {
                // Read in beta block-by-block when on the first row
                cb_reserve_back(cb_beta, block_size);
                DPRINT << "reserve_back beta on tile_row: " << tile_row << " col_tile: " << col_tile << ENDL();
                uint32_t l1_write_addr_b = get_write_ptr(cb_beta);
                for (uint32_t i = 0; i < block_size && col_tile + i < Wt; i++) {
                    uint64_t beta_noc_addr = get_noc_addr(col_tile + i, addrb);
                    async_read_row_to_tile<beta_is_row_major>(beta_noc_addr, l1_write_addr_b);
                    l1_write_addr_b += beta_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_beta, block_size);
            }
#endif
        }
    }
}
template <uint32_t t>
void async_read_row_to_tile(const uint64_t DRAM_src_addr, uint32_t L1_dst_addr) {
    noc_async_read(DRAM_src_addr, L1_dst_addr, 32 * 2);
    if constexpr (t == 0) {  // TILE layout
        noc_async_read(DRAM_src_addr + 512, L1_dst_addr + 512, 64);
    } else if constexpr (t == 1) {  // ROW_MAJOR layout
        noc_async_read_barrier();
        uint64_t noc_addr = get_noc_addr(L1_dst_addr + 32);
        noc_async_read(noc_addr, L1_dst_addr + 512, 64);
    } else {
        static_assert(t == 0 || t == 1, "Layout must be ROW_MAJOR (t==1) or TILE (t==0)");
    }
}
