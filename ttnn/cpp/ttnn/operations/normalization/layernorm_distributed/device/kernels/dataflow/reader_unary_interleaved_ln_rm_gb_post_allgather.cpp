// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs, per device statistics, and gamma, beta, epsilon from interleaved dram.
 */

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "debug/assert.h"

template <uint32_t t>
void async_read_row_to_tile(const uint64_t DRAM_src_addr, uint32_t L1_dst_addr);
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);     // Source address in dram
    const uint32_t NCHt = get_arg_val<uint32_t>(1);         // Number of NCH tiles
    // const uint32_t Wt = get_arg_val<uint32_t>(2);           // Width in tiles
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);  // Tile offset for this core
    const uint32_t stats_tile_offset =
        get_arg_val<uint32_t>(4);  // Tile offset for stats input; status input is two tiles wide and contains E(x) and
                                   // E(x^2) in the left most columns per tile.

    const uint32_t gamma_addr = get_arg_val<uint32_t>(7);
    const uint32_t beta_addr = get_arg_val<uint32_t>(8);
    const uint32_t stats_addr = get_arg_val<uint32_t>(9);
    const uint32_t y_offset = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_stats = tt::CBIndex::c_1;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta = tt::CBIndex::c_3;
    constexpr uint32_t cb_eps = tt::CBIndex::c_4;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_5;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);
    const uint32_t stats_tile_bytes = get_tile_size(cb_stats);

    constexpr uint32_t blk = get_compile_time_arg_val(0);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(1);
    constexpr uint32_t gamma_stick_size = get_compile_time_arg_val(2);
    constexpr uint32_t beta_stick_size = get_compile_time_arg_val(3);
    constexpr uint32_t gamma_is_row_major = get_compile_time_arg_val(4);
    constexpr uint32_t beta_is_row_major = get_compile_time_arg_val(5);
    constexpr uint32_t cb_length = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);  // Width in tiles
    constexpr auto src_args = TensorAccessorArgs<8>();
    constexpr auto stats_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const auto src_a = TensorAccessor(src_args, src_addr, src0_tile_bytes);
    const auto src_stats = TensorAccessor(stats_args, stats_addr, stats_tile_bytes);

#ifdef FUSE_GAMMA
    const auto addrg = TensorAccessor(gamma_args, gamma_addr, gamma_stick_size);
    const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
#endif
#ifdef FUSE_BETA
    const auto addrb = TensorAccessor(beta_args, beta_addr, beta_stick_size);
    const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
#endif

    // Generate constant tiles for layernorm compute
    uint32_t scaler = get_arg_val<uint32_t>(5);
    generate_reduce_scaler(cb_reduce, scaler);
    const uint32_t eps = get_arg_val<uint32_t>(6);
    generate_bcast_col_scalar(cb_eps, eps);

    uint32_t inp_tile_idx = tile_offset;
    uint32_t stats_tile_idx = stats_tile_offset;

    constexpr uint32_t cb_iterations = Wt / cb_length;
    constexpr uint32_t cb_leftovers = Wt % cb_length;
    constexpr uint32_t blk_iterations = cb_length / blk;
    constexpr uint32_t blk_leftovers = cb_length % blk;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Read stats tiles
        cb_reserve_back(cb_stats, stats_tiles_cols);
        uint32_t stats_wr_ptr = get_write_ptr(cb_stats);
        for (uint32_t st = 0; st < stats_tiles_cols; ++st) {
            noc_async_read_tile(stats_tile_idx, src_stats, stats_wr_ptr);
            stats_wr_ptr += stats_tile_bytes;
            stats_tile_idx++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_stats, stats_tiles_cols);
        uint32_t gamma_tile_count = 0;
        uint32_t beta_tile_count = 0;
        for (uint32_t i = 0; i < cb_iterations; i++) {
            for (uint32_t j = 0; j < cb_length; j++) {
                cb_reserve_back(cb_inp, 1);
                uint32_t inp_wr_ptr = get_write_ptr(cb_inp);
                noc_async_read_tile(inp_tile_idx, src_a, inp_wr_ptr);
                inp_tile_idx++;
                noc_async_read_barrier();
                cb_push_back(cb_inp, 1);
            }
#if defined FUSE_GAMMA || defined FUSE_BETA
#ifdef FUSE_GAMMA
            for (uint32_t j = 0; j < cb_length; j++) {
                cb_reserve_back(cb_gamma, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_gamma);
                uint64_t gamma_noc_addr = get_noc_addr(gamma_tile_count, addrg);
                gamma_tile_count++;
                async_read_row_to_tile<gamma_is_row_major>(gamma_noc_addr, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_gamma, 1);
            }
#endif
#ifdef FUSE_BETA
            for (uint32_t j = 0; j < cb_length; j++) {
                cb_reserve_back(cb_beta, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_beta);
                uint64_t beta_noc_addr = get_noc_addr(beta_tile_count, addrb);
                beta_tile_count++;
                async_read_row_to_tile<beta_is_row_major>(beta_noc_addr, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_beta, 1);
            }
#endif
#endif
        }
        for (uint32_t i = 0; i < cb_leftovers; i++) {
            cb_reserve_back(cb_inp, 1);
            uint32_t inp_wr_ptr = get_write_ptr(cb_inp);
            noc_async_read_tile(inp_tile_idx, src_a, inp_wr_ptr);
            inp_tile_idx++;
            noc_async_read_barrier();
            cb_push_back(cb_inp, 1);
        }
#if defined FUSE_GAMMA || defined FUSE_BETA
#ifdef FUSE_GAMMA
        for (uint32_t i = 0; i < cb_leftovers; i++) {
            cb_reserve_back(cb_gamma, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_gamma);
            uint64_t gamma_noc_addr = get_noc_addr(gamma_tile_count, addrg);
            gamma_tile_count++;
            async_read_row_to_tile<gamma_is_row_major>(gamma_noc_addr, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_gamma, 1);
        }
#endif
#ifdef FUSE_BETA
        for (uint32_t i = 0; i < cb_leftovers; i++) {
            cb_reserve_back(cb_beta, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_beta);
            uint64_t beta_noc_addr = get_noc_addr(beta_tile_count, addrb);
            beta_tile_count++;
            async_read_row_to_tile<beta_is_row_major>(beta_noc_addr, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_beta, 1);
        }
#endif
#endif
    }  // ncht loop
}
template <uint32_t t>
void async_read_row_to_tile(const uint64_t DRAM_src_addr, uint32_t L1_dst_addr) {
    noc_async_read(DRAM_src_addr, L1_dst_addr, 32 * 2);  // reads 32 elements (64 bytes) 16 usefull, the next bad
    if constexpr (t == 0) {  // TILE LAYOUT
        noc_async_read(DRAM_src_addr + 512, L1_dst_addr + 512, 64);  // Fills the second face with next 16 elements
    } else if constexpr (t == 1) {  // ROW MAJOR LAYOUT
        noc_async_read_barrier();
        uint64_t noc_addr = get_noc_addr(L1_dst_addr + 32);  // 16 elements from DRAM to L1.  L1->L1
        noc_async_read(noc_addr, L1_dst_addr + 512, 64);
    } else {
        static_assert(false, "Layout must be ROW_MAJOR(t == 1) or TILE_LAYOUT(t == 0)");
    }
}
