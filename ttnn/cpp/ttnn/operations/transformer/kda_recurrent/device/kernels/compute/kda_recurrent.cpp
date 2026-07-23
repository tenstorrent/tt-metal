// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q = 0;
constexpr uint32_t cb_k = 1;
constexpr uint32_t cb_v = 2;
constexpr uint32_t cb_decay = 3;
constexpr uint32_t cb_beta = 4;
constexpr uint32_t cb_state = 5;
constexpr uint32_t cb_decayed_state = 6;
constexpr uint32_t cb_v_read = 7;
constexpr uint32_t cb_delta = 8;
constexpr uint32_t cb_beta_delta = 9;
constexpr uint32_t cb_k_column = 10;
constexpr uint32_t cb_update = 11;
constexpr uint32_t cb_updated_state = 12;
constexpr uint32_t cb_output = 13;
constexpr uint32_t cb_final_state = 14;

inline void wait(uint32_t cb, uint32_t tiles) { CircularBuffer(cb).wait_front(tiles); }
inline void pop(uint32_t cb, uint32_t tiles) { CircularBuffer(cb).pop_front(tiles); }

void matrix_multiply(uint32_t left, uint32_t right, uint32_t output, uint32_t rows, uint32_t inner, uint32_t columns) {
    cb_reserve_back(output, rows * columns);
    pack_reconfig_data_format(output);
    reconfig_data_format(right, left);
    matmul_init(left, right, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            tile_regs_acquire();
            for (uint32_t index = 0; index < inner; ++index) {
                matmul_tiles(left, right, row * inner + index, index * columns + column, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, output, row * columns + column);
            tile_regs_release();
        }
    }
    cb_push_back(output, rows * columns);
}

void elementwise(uint32_t left, uint32_t right, uint32_t output, uint32_t tiles, bool subtract) {
    cb_reserve_back(output, tiles);
    pack_reconfig_data_format(output);
    reconfig_data_format(left, right);
    if (subtract) {
        sub_tiles_init(left, right);
    } else {
        add_tiles_init(left, right);
    }
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        if (subtract) {
            sub_tiles(left, right, tile, tile, 0);
        } else {
            add_tiles(left, right, tile, tile, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output, tile);
        tile_regs_release();
    }
    cb_push_back(output, tiles);
}

void multiply_by_column(uint32_t matrix, uint32_t column, uint32_t output, uint32_t rows, uint32_t columns) {
    cb_reserve_back(output, rows * columns);
    pack_reconfig_data_format(output);
    reconfig_data_format(matrix, column);
    mul_bcast_cols_init_short(matrix, column);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < columns; ++col) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(matrix, column, row * columns + col, row, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, output, row * columns + col);
            tile_regs_release();
        }
    }
    cb_push_back(output, rows * columns);
}

void multiply_by_scalar(uint32_t input, uint32_t scalar, uint32_t output, uint32_t tiles) {
    cb_reserve_back(output, tiles);
    pack_reconfig_data_format(output);
    reconfig_data_format(input, scalar);
    mul_tiles_bcast_scalar_init_short(input, scalar);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(input, scalar, tile, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output, tile);
        tile_regs_release();
    }
    cb_push_back(output, tiles);
}

void transpose_row(uint32_t input, uint32_t output, uint32_t tiles) {
    cb_reserve_back(output, tiles);
    pack_reconfig_data_format(output);
    reconfig_data_format_srca(input);
    transpose_init(input);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        transpose_tile(input, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output, tile);
        tile_regs_release();
    }
    cb_push_back(output, tiles);
}

void copy_tiles(uint32_t input, uint32_t output, uint32_t tiles) {
    cb_reserve_back(output, tiles);
    pack_reconfig_data_format(output);
    reconfig_data_format_srca(input);
    copy_tile_to_dst_init_short(input);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        copy_tile(input, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output, tile);
        tile_regs_release();
    }
    cb_push_back(output, tiles);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t key_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t value_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t state_tiles = key_tiles * value_tiles;

    compute_kernel_hw_startup(cb_q, cb_k, cb_output);
    wait(cb_q, key_tiles);
    wait(cb_k, key_tiles);
    wait(cb_v, value_tiles);
    wait(cb_decay, key_tiles);
    wait(cb_beta, 1);
    wait(cb_state, state_tiles);

    // S_decay[k,v] = decay[k] * S[k,v].
    multiply_by_column(cb_state, cb_decay, cb_decayed_state, key_tiles, value_tiles);
    wait(cb_decayed_state, state_tiles);
    pop(cb_state, state_tiles);
    pop(cb_decay, key_tiles);

    // delta = beta * (v - k @ S_decay).
    matrix_multiply(cb_k, cb_decayed_state, cb_v_read, 1, key_tiles, value_tiles);
    wait(cb_v_read, value_tiles);
    elementwise(cb_v, cb_v_read, cb_delta, value_tiles, true);
    wait(cb_delta, value_tiles);
    pop(cb_v, value_tiles);
    pop(cb_v_read, value_tiles);
    multiply_by_scalar(cb_delta, cb_beta, cb_beta_delta, value_tiles);
    wait(cb_beta_delta, value_tiles);
    pop(cb_delta, value_tiles);
    pop(cb_beta, 1);

    // S_new = S_decay + k.T @ delta.
    transpose_row(cb_k, cb_k_column, key_tiles);
    wait(cb_k_column, key_tiles);
    pop(cb_k, key_tiles);
    matrix_multiply(cb_k_column, cb_beta_delta, cb_update, key_tiles, 1, value_tiles);
    wait(cb_update, state_tiles);
    pop(cb_k_column, key_tiles);
    pop(cb_beta_delta, value_tiles);
    elementwise(cb_decayed_state, cb_update, cb_updated_state, state_tiles, false);
    wait(cb_updated_state, state_tiles);
    pop(cb_decayed_state, state_tiles);
    pop(cb_update, state_tiles);

    // Query observes the updated state.
    matrix_multiply(cb_q, cb_updated_state, cb_output, 1, key_tiles, value_tiles);
    wait(cb_output, value_tiles);
    pop(cb_q, key_tiles);

    // The writer owns a distinct CB; updated_state remains a compute input above.
    copy_tiles(cb_updated_state, cb_final_state, state_tiles);
    pop(cb_updated_state, state_tiles);
}
