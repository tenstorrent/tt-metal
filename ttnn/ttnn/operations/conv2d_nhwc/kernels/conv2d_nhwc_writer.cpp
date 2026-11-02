// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc writer (BRISC, NoC1) — regime R0.
//
// Block operations owned here:
//   load_bias_slice      : core_n_tiles bias tiles, once per core (compute keeps them fronted).
//   weight K-block fill  : per (m-block, n-block), num_k_blocks K-blocks of bk x bn weight tiles
//                          (row-major [k][n] in cb_weight_tiles), double-buffered.
//   untilize_store_block : per tile-row, 32 RM rows of bn*32 output channels written to DRAM
//                          at byte offset n_start*32*elem_bytes of each pixel's output stick;
//                          rows >= M or past the core's m range are dropped.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_weight_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_bias_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_out_rm = get_compile_time_arg_val(2);
constexpr uint32_t block_m_tiles = get_compile_time_arg_val(3);
constexpr uint32_t block_k_tiles = get_compile_time_arg_val(4);
constexpr uint32_t block_n_tiles = get_compile_time_arg_val(5);
constexpr uint32_t num_k_blocks = get_compile_time_arg_val(6);
constexpr uint32_t num_n_blocks = get_compile_time_arg_val(7);
constexpr uint32_t N_t = get_compile_time_arg_val(8);
constexpr uint32_t core_n_tiles = get_compile_time_arg_val(9);
constexpr uint32_t has_bias = get_compile_time_arg_val(10);
constexpr uint32_t M = get_compile_time_arg_val(11);
constexpr uint32_t elem_bytes = get_compile_time_arg_val(12);
constexpr auto weight_args = TensorAccessorArgs<13>();
[[maybe_unused]] constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
constexpr auto output_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t out_row_bytes = block_n_tiles * TILE_DIM * elem_bytes;

void kernel_main() {
    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t m_tile_start = get_arg_val<uint32_t>(3);
    const uint32_t m_tiles_this_core = get_arg_val<uint32_t>(4);
    const uint32_t num_m_blocks = get_arg_val<uint32_t>(5);
    const uint32_t n_tile_start = get_arg_val<uint32_t>(6);

    const auto weights = TensorAccessor(weight_args, weight_addr);
    const auto output = TensorAccessor(output_args, output_addr);
    const uint32_t weight_tile_bytes = get_tile_size(cb_weight_tiles);

    // ---- load_bias_slice ----
    if constexpr (has_bias) {
        const auto bias = TensorAccessor(bias_args, bias_addr);
        const uint32_t bias_tile_bytes = get_tile_size(cb_bias_tiles);
        cb_reserve_back(cb_bias_tiles, core_n_tiles);
        uint32_t dst = get_write_ptr(cb_bias_tiles);
        for (uint32_t j = 0; j < core_n_tiles; ++j) {
            noc_async_read(bias.get_noc_addr(n_tile_start + j), dst, bias_tile_bytes);
            dst += bias_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_bias_tiles, core_n_tiles);
    }

    const uint32_t m_tile_end = m_tile_start + m_tiles_this_core;
    for (uint32_t m_block = 0; m_block < num_m_blocks; ++m_block) {
        const uint32_t t0 = m_tile_start + m_block * block_m_tiles;
        const uint32_t t1 = (t0 + block_m_tiles < m_tile_end) ? t0 + block_m_tiles : m_tile_end;
        for (uint32_t n_block = 0; n_block < num_n_blocks; ++n_block) {
            const uint32_t n0 = n_tile_start + n_block * block_n_tiles;

            // ---- weight K-blocks for this (m, n) block ----
            for (uint32_t k_block = 0; k_block < num_k_blocks; ++k_block) {
                cb_reserve_back(cb_weight_tiles, block_k_tiles * block_n_tiles);
                uint32_t dst = get_write_ptr(cb_weight_tiles);
                uint32_t row_page = k_block * block_k_tiles * N_t + n0;
                for (uint32_t kk = 0; kk < block_k_tiles; ++kk) {
                    for (uint32_t j = 0; j < block_n_tiles; ++j) {
                        noc_async_read(weights.get_noc_addr(row_page + j), dst, weight_tile_bytes);
                        dst += weight_tile_bytes;
                    }
                    row_page += N_t;
                }
                noc_async_read_barrier();
                cb_push_back(cb_weight_tiles, block_k_tiles * block_n_tiles);
            }

            // ---- untilize_store_block: drain bm tile-rows ----
            const uint32_t col_offset_bytes = n0 * TILE_DIM * elem_bytes;
            for (uint32_t t = 0; t < block_m_tiles; ++t) {
                cb_wait_front(cb_out_rm, block_n_tiles);
                const uint32_t tile = t0 + t;
                if (tile < t1) {
                    uint32_t src = get_read_ptr(cb_out_rm);
                    uint32_t pix = tile * TILE_DIM;
                    for (uint32_t r = 0; r < TILE_DIM && pix < M; ++r, ++pix) {
                        noc_async_write(src, output.get_noc_addr(pix, col_offset_bytes), out_row_bytes);
                        src += out_row_bytes;
                    }
                    noc_async_write_barrier();
                }
                cb_pop_front(cb_out_rm, block_n_tiles);
            }
        }
    }
}
