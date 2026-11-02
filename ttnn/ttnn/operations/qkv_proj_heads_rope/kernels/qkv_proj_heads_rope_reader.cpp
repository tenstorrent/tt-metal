// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// qkv_proj_heads_rope reader (NCRISC): x K-blocks (+ the synthesized ones-column bias tile) into
// cb_activation_block, then per valid row the rotary table tiles [cos_lo, -sin_lo, sin_hi, cos_hi].
// The ones tile and the sin sign flip are tile-content transforms no dataflow helper provides.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t FACE_ELEMS = 16 * 16;
constexpr uint16_t BF16_ONE = 0x3F80;

// Column 0 = 1.0 in all 32 rows (faces 0 and 2); the rest of the tile is pre-zeroed.
FORCE_INLINE void write_ones_column(uint32_t tile_addr) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr);
    for (uint32_t row = 0; row < 16; ++row) {
        tile[row * 16] = BF16_ONE;
        tile[2 * FACE_ELEMS + row * 16] = BF16_ONE;
    }
}

template <uint32_t tile_bytes>
FORCE_INLINE void negate_bf16_tile(uint32_t tile_addr) {
    // Plain (non-volatile) pointer: the page is owned by this RISC between the read barrier and the push,
    // so the compiler may batch the loads.
    uint32_t* words = reinterpret_cast<uint32_t*>(tile_addr);
    constexpr uint32_t num_words = tile_bytes / sizeof(uint32_t);
    constexpr uint32_t unroll = 8;
    static_assert(num_words % unroll == 0);
    for (uint32_t idx = 0; idx < num_words; idx += unroll) {
        uint32_t vals[unroll];
#pragma GCC unroll 8
        for (uint32_t u = 0; u < unroll; ++u) {
            vals[u] = words[idx + u];
        }
#pragma GCC unroll 8
        for (uint32_t u = 0; u < unroll; ++u) {
            words[idx + u] = vals[u] ^ 0x80008000u;
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_activation_block = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rotary_table = get_compile_time_arg_val(1);
    constexpr uint32_t block_m = get_compile_time_arg_val(2);
    constexpr uint32_t k_block = get_compile_time_arg_val(3);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(8);
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto cos_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_start = get_arg_val<uint32_t>(3);
    const uint32_t core_rows = get_arg_val<uint32_t>(4);
    const bool is_rotary = get_arg_val<uint32_t>(5) != 0;
    // Cores sharing x rows (one M-group) start each K-block at different tiles so they don't hit one bank in lockstep.
    const uint32_t stagger_idx = get_arg_val<uint32_t>(6);
    const uint32_t stagger_count = get_arg_val<uint32_t>(7);

    // K' = hidden tiles + the bias K tile.
    static_assert(num_k_blocks * k_block == hidden_tiles + 1);
    constexpr uint32_t act_block_tiles = block_m * k_block;
    // Table row = [cos_lo, -sin_lo, sin_hi, cos_hi]: the rotate_half tile swap needs D/2 == one tile.
    static_assert(tiles_per_head == 2);
    constexpr uint32_t table_tiles_per_row = 2 * tiles_per_head;

    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto cos_accessor = TensorAccessor(cos_args, cos_addr, tile_bytes);
    const auto sin_accessor = TensorAccessor(sin_args, sin_addr, tile_bytes);

    Noc noc;
    CircularBuffer activation(cb_activation_block);

    for (uint32_t block_row = 0; block_row < core_rows; block_row += block_m) {
        const uint32_t rows = (core_rows - block_row) < block_m ? (core_rows - block_row) : block_m;
        const uint32_t first_row = row_start + block_row;

        // Nominal block quantum keeps the ring aligned; rows past `rows` stay stale and are never consumed.
        for (uint32_t k_blk = 0; k_blk < num_k_blocks; ++k_blk) {
            const uint32_t k_first = k_blk * k_block;
            const bool has_ones = k_first + k_block > hidden_tiles;
            const uint32_t ones_col = hidden_tiles - k_first;
            const uint32_t x_cols = has_ones ? ones_col : k_block;

            cb_reserve_back(cb_activation_block, act_block_tiles);
            const uint32_t block_addr = get_write_ptr(cb_activation_block);
            if (has_ones) {
                for (uint32_t row = 0; row < rows; ++row) {
                    noc.async_write_zeros(
                        activation, tile_bytes, {.offset_bytes = (row * k_block + ones_col) * tile_bytes});
                }
            }
            const uint32_t x_tiles = rows * x_cols;
            const uint32_t start = stagger_idx * x_tiles / stagger_count;
            uint32_t row = start / x_cols;
            uint32_t col = start % x_cols;
            for (uint32_t tile = 0; tile < x_tiles; ++tile) {
                noc_async_read_page(
                    (first_row + row) * hidden_tiles + k_first + col,
                    x_accessor,
                    block_addr + (row * k_block + col) * tile_bytes);
                if (++col == x_cols) {
                    col = 0;
                    if (++row == rows) {
                        row = 0;
                    }
                }
            }
            noc_async_read_barrier();
            if (has_ones) {
                noc.write_zeros_l1_barrier();
                for (uint32_t row = 0; row < rows; ++row) {
                    write_ones_column(block_addr + (row * k_block + ones_col) * tile_bytes);
                }
            }
            cb_push_back(cb_activation_block, act_block_tiles);
        }

        if (is_rotary) {
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t st = (first_row + row) % seq_tiles;
                cb_reserve_back(cb_rotary_table, table_tiles_per_row);
                const uint32_t table_addr = get_write_ptr(cb_rotary_table);
                const uint32_t lo_page = st * tiles_per_head;
                noc_async_read_page(lo_page, cos_accessor, table_addr);
                noc_async_read_page(lo_page, sin_accessor, table_addr + tile_bytes);
                noc_async_read_page(lo_page + 1, sin_accessor, table_addr + 2 * tile_bytes);
                noc_async_read_page(lo_page + 1, cos_accessor, table_addr + 3 * tile_bytes);
                noc_async_read_barrier();
                negate_bf16_tile<tile_bytes>(table_addr + tile_bytes);
                cb_push_back(cb_rotary_table, table_tiles_per_row);
            }
        }
    }
}
