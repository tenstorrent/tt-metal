// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-based compute kernel for YUV conversion.
//
// Processes 32 spatial positions per tile (full tile utilization).
// Reader fills row-major bf16 CBs (32 sticks × 32 columns per tile).
// This kernel tilizes, does linear combination with scalar broadcast,
// clamps [0, 255], untilizes back to row-major bf16, and pushes to output CB.
//
// Three sequential phases (Y, Cb, Cr), each with its own scalar coefficients.
// The scalar tile CBs are populated by the reader and popped here between phases.
//
// Y tile:  tilize(R)*wr + tilize(G)*wg + tilize(B)*wb + off → clamp → untilize
// UV tile: (sum of 4 corners per channel, each multiplied by weight) + off → clamp
//          Reader pre-scales UV weights by 0.25, so sum*w*0.25 = avg*w.
//
// Compile-time args:
//   [0]  cb_R_rm     [1]  cb_G_rm     [2]  cb_B_rm
//   [3]  cb_tilized  [4]  cb_partial  [5]  cb_temp
//   [6]  cb_wr       [7]  cb_wg       [8]  cb_wb       [9]  cb_off
//   [10] cb_sum      [11] cb_out_rm
//   [12] num_y_tiles           — y_batches * num_t_tiles
//   [13] num_uv_tiles_per_plane — uv_batches * num_t_tiles
//   [14] num_t_tiles

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "api/compute/eltwise_unary/clamp.h"

static constexpr uint32_t BF16_ZERO_BITS = 0x00000000u;  // 0.0f
static constexpr uint32_t BF16_255_BITS = 0x437F0000u;   // 255.0f

// Tilize one row-major bf16 CB page, multiply by scalar tile, pack result.
FORCE_INLINE void tilize_and_mul_scalar(uint32_t cb_rm, uint32_t cb_tilized, uint32_t cb_scalar, uint32_t cb_dst) {
    tilize_init(cb_rm, 1, cb_tilized);
    cb_wait_front(cb_rm, 1);
    cb_reserve_back(cb_tilized, 1);
    tilize_block(cb_rm, 1, cb_tilized);
    cb_push_back(cb_tilized, 1);
    cb_pop_front(cb_rm, 1);
    tilize_uninit(cb_rm, cb_tilized);

    tile_regs_acquire();
    cb_wait_front(cb_tilized, 1);
    mul_tiles_bcast_scalar_init_short(cb_tilized, cb_scalar);
    mul_tiles_bcast_scalar(cb_tilized, cb_scalar, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_dst, 1);
    pack_tile(0, cb_dst);
    cb_push_back(cb_dst, 1);
    tile_regs_release();
    cb_pop_front(cb_tilized, 1);
}

// Add two tile CBs, pack result.
FORCE_INLINE void add_and_pack(uint32_t cb_a, uint32_t cb_b, uint32_t cb_dst) {
    tile_regs_acquire();
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    add_tiles_init(cb_a, cb_b);
    add_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_dst, 1);
    pack_tile(0, cb_dst);
    cb_push_back(cb_dst, 1);
    tile_regs_release();
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}

// Add scalar offset, clamp [0,255], pack to tile CB, untilize to row-major.
FORCE_INLINE void offset_clamp_untilize(uint32_t cb_in, uint32_t cb_off, uint32_t cb_sum, uint32_t cb_out_rm) {
    tile_regs_acquire();
    cb_wait_front(cb_in, 1);
    add_bcast_scalar_init_short(cb_in, cb_off);
    add_tiles_bcast_scalar(cb_in, cb_off, 0, 0, 0);
    clamp_tile_init();
    clamp_tile(0, BF16_ZERO_BITS, BF16_255_BITS);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_sum, 1);
    pack_tile(0, cb_sum);
    cb_push_back(cb_sum, 1);
    tile_regs_release();
    cb_pop_front(cb_in, 1);

    untilize_init(cb_sum);
    cb_wait_front(cb_sum, 1);
    cb_reserve_back(cb_out_rm, 1);
    untilize_block(cb_sum, 1, cb_out_rm);
    cb_push_back(cb_out_rm, 1);
    cb_pop_front(cb_sum, 1);
    untilize_uninit(cb_sum);
}

void kernel_main() {
    constexpr uint32_t cb_R_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_G_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_B_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tilized = get_compile_time_arg_val(3);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(4);
    constexpr uint32_t cb_temp = get_compile_time_arg_val(5);
    constexpr uint32_t cb_wr = get_compile_time_arg_val(6);
    constexpr uint32_t cb_wg = get_compile_time_arg_val(7);
    constexpr uint32_t cb_wb = get_compile_time_arg_val(8);
    constexpr uint32_t cb_off = get_compile_time_arg_val(9);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(10);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(11);
    constexpr uint32_t num_y_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t num_uv_tiles_per_plane = get_compile_time_arg_val(13);
    constexpr uint32_t num_t_tiles = get_compile_time_arg_val(14);

    compute_kernel_hw_startup(cb_R_rm, cb_tilized);

    // ---- Phase 1: Y pass ----
    // Scalar tiles already pushed by reader.
    cb_wait_front(cb_wr, 1);
    cb_wait_front(cb_wg, 1);
    cb_wait_front(cb_wb, 1);
    cb_wait_front(cb_off, 1);

    for (uint32_t t = 0; t < num_y_tiles; t++) {
        // Y = wr*R + wg*G + wb*B + offset, clamped [0,255]
        tilize_and_mul_scalar(cb_R_rm, cb_tilized, cb_wr, cb_partial);
        tilize_and_mul_scalar(cb_G_rm, cb_tilized, cb_wg, cb_temp);
        add_and_pack(cb_partial, cb_temp, cb_partial);
        tilize_and_mul_scalar(cb_B_rm, cb_tilized, cb_wb, cb_temp);
        add_and_pack(cb_partial, cb_temp, cb_partial);
        offset_clamp_untilize(cb_partial, cb_off, cb_sum, cb_out_rm);
    }

    // Pop Y scalars so reader can push Cb/Cr scalars.
    cb_pop_front(cb_wr, 1);
    cb_pop_front(cb_wg, 1);
    cb_pop_front(cb_wb, 1);
    cb_pop_front(cb_off, 1);

    // ---- Phases 2 & 3: Cb and Cr passes ----
    for (uint32_t plane = 0; plane < 2; plane++) {
        // Wait for new scalar tiles from reader.
        cb_wait_front(cb_wr, 1);
        cb_wait_front(cb_wg, 1);
        cb_wait_front(cb_wb, 1);
        cb_wait_front(cb_off, 1);

        for (uint32_t t = 0; t < num_uv_tiles_per_plane; t++) {
            // UV = sum_corners(wr_scaled * R) + sum_corners(wg_scaled * G)
            //    + sum_corners(wb_scaled * B) + offset
            // Where wr_scaled = wr * 0.25 (reader pre-scales the scalar).
            //
            // Reader sends 4 R corners, then 4 G corners, then 4 B corners.
            // For each corner page: tilize → multiply by channel weight → add to accumulator.
            // This avoids needing extra CBs for corner accumulation.

            // R corner 0 → cb_partial
            tilize_and_mul_scalar(cb_R_rm, cb_tilized, cb_wr, cb_partial);
            // R corners 1-3
            for (uint32_t c = 1; c < 4; c++) {
                tilize_and_mul_scalar(cb_R_rm, cb_tilized, cb_wr, cb_temp);
                add_and_pack(cb_partial, cb_temp, cb_partial);
            }

            // G corners 0-3: multiply by wg, add to accumulator
            for (uint32_t c = 0; c < 4; c++) {
                tilize_and_mul_scalar(cb_G_rm, cb_tilized, cb_wg, cb_temp);
                add_and_pack(cb_partial, cb_temp, cb_partial);
            }

            // B corners 0-3: multiply by wb, add to accumulator
            for (uint32_t c = 0; c < 4; c++) {
                tilize_and_mul_scalar(cb_B_rm, cb_tilized, cb_wb, cb_temp);
                add_and_pack(cb_partial, cb_temp, cb_partial);
            }

            // + offset, clamp, untilize → cb_out_rm
            offset_clamp_untilize(cb_partial, cb_off, cb_sum, cb_out_rm);
        }

        cb_pop_front(cb_wr, 1);
        cb_pop_front(cb_wg, 1);
        cb_pop_front(cb_wb, 1);
        cb_pop_front(cb_off, 1);
    }
}
