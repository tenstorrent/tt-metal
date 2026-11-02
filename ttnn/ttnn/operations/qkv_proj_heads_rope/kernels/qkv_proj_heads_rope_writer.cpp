// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// qkv_proj_heads_rope writer (BRISC): streams this core's W column slice (+ the bias tile row as the
// last K row) into cb_weight_block, then scatters each valid output row into q|k|v head layout.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_weight_block = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_m = get_compile_time_arg_val(2);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t k_block = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t weight_width_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_heads = get_compile_time_arg_val(9);
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(10);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(11);
    constexpr auto weight_args = TensorAccessorArgs<12>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t third = get_arg_val<uint32_t>(5);
    const uint32_t first_col_tile = get_arg_val<uint32_t>(6);
    const uint32_t first_head = get_arg_val<uint32_t>(7);
    const uint32_t row_start = get_arg_val<uint32_t>(8);
    const uint32_t core_rows = get_arg_val<uint32_t>(9);
    // Cores sharing a W slice (one N-group) start each K-block at different tiles so they don't read in lockstep.
    const uint32_t stagger_idx = get_arg_val<uint32_t>(10);
    const uint32_t stagger_count = get_arg_val<uint32_t>(11);

    static_assert(num_k_blocks * k_block == hidden_tiles + 1);
    constexpr uint32_t weight_block_tiles = k_block * n_tiles;
    // Bias tile rows 1..31 (padding, possibly non-finite) are zeroed so 0 * pad cannot poison the fold:
    // face 0/1 rows 1..15 and faces 2/3 entirely.
    constexpr uint32_t face_row_bytes = 16 * 2;
    constexpr uint32_t face_bytes = 16 * face_row_bytes;

    const auto weight_accessor = TensorAccessor(weight_args, weight_addr, tile_bytes);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr, tile_bytes);
    const auto q_accessor = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);

    Noc noc;
    CircularBuffer weight_block(cb_weight_block);

    for (uint32_t block_row = 0; block_row < core_rows; block_row += block_m) {
        const uint32_t rows = (core_rows - block_row) < block_m ? (core_rows - block_row) : block_m;

        for (uint32_t k_blk = 0; k_blk < num_k_blocks; ++k_blk) {
            cb_reserve_back(cb_weight_block, weight_block_tiles);
            const uint32_t block_addr = get_write_ptr(cb_weight_block);
            const uint32_t start = stagger_idx * weight_block_tiles / stagger_count;
            uint32_t k = start / n_tiles;
            uint32_t n = start % n_tiles;
            for (uint32_t tile = 0; tile < weight_block_tiles; ++tile) {
                const uint32_t kt = k_blk * k_block + k;
                const uint32_t l1_addr = block_addr + (k * n_tiles + n) * tile_bytes;
                if (kt < hidden_tiles) {
                    noc_async_read_page(kt * weight_width_tiles + first_col_tile + n, weight_accessor, l1_addr);
                } else {
                    noc_async_read_page(first_col_tile + n, bias_accessor, l1_addr);
                }
                if (++n == n_tiles) {
                    n = 0;
                    if (++k == k_block) {
                        k = 0;
                    }
                }
            }
            const uint32_t bias_k = hidden_tiles >= k_blk * k_block ? hidden_tiles - k_blk * k_block : k_block;
            noc_async_read_barrier();
            if (bias_k < k_block) {
                for (uint32_t n = 0; n < n_tiles; ++n) {
                    const uint32_t tile_off = (bias_k * n_tiles + n) * tile_bytes;
                    noc.async_write_zeros(
                        weight_block, face_bytes - face_row_bytes, {.offset_bytes = tile_off + face_row_bytes});
                    noc.async_write_zeros(
                        weight_block,
                        face_bytes - face_row_bytes,
                        {.offset_bytes = tile_off + face_bytes + face_row_bytes});
                    noc.async_write_zeros(weight_block, 2 * face_bytes, {.offset_bytes = tile_off + 2 * face_bytes});
                }
                noc.write_zeros_l1_barrier();
            }
            cb_push_back(cb_weight_block, weight_block_tiles);
        }

        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t r = row_start + block_row + row;
            const uint32_t batch = r / seq_tiles;
            const uint32_t st = r % seq_tiles;
            cb_wait_front(cb_out_tiles, n_tiles);
            uint32_t l1_addr = get_read_ptr(cb_out_tiles);
            for (uint32_t n = 0; n < n_tiles; ++n) {
                const uint32_t head = first_head + n / tiles_per_head;
                const uint32_t page =
                    ((batch * num_heads + head) * seq_tiles + st) * tiles_per_head + n % tiles_per_head;
                if (third == 0) {
                    noc_async_write_page(page, q_accessor, l1_addr);
                } else if (third == 1) {
                    noc_async_write_page(page, k_accessor, l1_addr);
                } else {
                    noc_async_write_page(page, v_accessor, l1_addr);
                }
                l1_addr += tile_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out_tiles, n_tiles);
        }
    }
    noc_async_write_barrier();
}
