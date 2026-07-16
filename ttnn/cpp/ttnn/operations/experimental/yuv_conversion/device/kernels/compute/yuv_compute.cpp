// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-based compute kernel for YUV conversion (per-unit fused Y/Cb/Cr).
//
// The reader feeds one (row-group, T-tile) unit at a time: y_tiles worth of
// flat RGB tiles for Y, then uv_tiles worth of 4-corner RGB tiles for Cb, then
// the same for Cr.  Because each unit is identical in shape, a single compute
// kernel serves all cores; the per-core unit count is a runtime arg.
//
// Per tile: tilize, linear-combine with a broadcast scalar, clamp [0,255],
// untilize back to row-major bf16, then SFPU typecast bf16->uint8.
//
// The 12 scalar coefficient tiles (Y/Cb/Cr x wr/wg/wb/off) are populated once
// by the reader and kept resident here (waited on once, never popped).
//
// Compile-time args:
//   [0] cb_R_rm     [1] cb_G_rm     [2] cb_B_rm
//   [3] cb_tilized  [4] cb_partial  [5] cb_temp   [6] cb_sum
//   [7] cb_out_bf16 [8] cb_out_rm   [9] cb_scalar_base (12 resident scalar CBs)
//   [10] y_tiles    [11] uv_tiles
//
// Runtime args:
//   [0] unit_count

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace tc = compute_kernel_lib::tilize_config;
namespace uc = compute_kernel_lib::untilize_config;

static constexpr uint32_t BF16_ZERO_BITS = 0x00000000u;  // 0.0f
static constexpr uint32_t BF16_255_BITS = 0x437F0000u;   // 255.0f

FORCE_INLINE void mul_scalar_pack(uint32_t cb_tilized, uint32_t cb_scalar, uint32_t cb_dst) {
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

FORCE_INLINE void offset_clamp_pack(uint32_t cb_in, uint32_t cb_off, uint32_t cb_sum) {
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
}

// SFPU typecast: bf16 -> uint8 on a row-major page (32x32 elements).
FORCE_INLINE void typecast_and_pack(uint32_t cb_bf16, uint32_t cb_u8) {
#ifdef ARCH_BLACKHOLE
    MATH((llk_math_reconfig_remap(false)));
#endif
    init_sfpu(cb_bf16, cb_u8);
    tile_regs_acquire();
    cb_wait_front(cb_bf16, 1);
    copy_tile(cb_bf16, 0, 0);
    TYPECAST_LLK_INIT();
    TYPECAST_LLK(0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_u8, 1);
    pack_tile(0, cb_u8);
    cb_push_back(cb_u8, 1);
    cb_pop_front(cb_bf16, 1);
    tile_regs_release();
}

template <uint32_t cb_src, uint32_t cb_tilized>
FORCE_INLINE void tilize_one() {
    compute_kernel_lib::tilize<
        1,
        cb_src,
        cb_tilized,
        tc::InitUninitMode::InitAndUninit,
        tc::WaitMode::WaitBlock,
        tc::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
}

void kernel_main() {
    constexpr uint32_t cb_R_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_G_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_B_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tilized = get_compile_time_arg_val(3);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(4);
    constexpr uint32_t cb_temp = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out_bf16 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(8);
    constexpr uint32_t cb_scalar_base = get_compile_time_arg_val(9);
    constexpr uint32_t y_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t uv_tiles = get_compile_time_arg_val(11);

    // Resident scalar CBs: (Y, Cb, Cr) x (wr, wg, wb, off).
    constexpr uint32_t cb_wr_y = cb_scalar_base + 0;
    constexpr uint32_t cb_wg_y = cb_scalar_base + 1;
    constexpr uint32_t cb_wb_y = cb_scalar_base + 2;
    constexpr uint32_t cb_off_y = cb_scalar_base + 3;
    constexpr uint32_t cb_wr_cb = cb_scalar_base + 4;
    constexpr uint32_t cb_wg_cb = cb_scalar_base + 5;
    constexpr uint32_t cb_wb_cb = cb_scalar_base + 6;
    constexpr uint32_t cb_off_cb = cb_scalar_base + 7;
    constexpr uint32_t cb_wr_cr = cb_scalar_base + 8;
    constexpr uint32_t cb_wg_cr = cb_scalar_base + 9;
    constexpr uint32_t cb_wb_cr = cb_scalar_base + 10;
    constexpr uint32_t cb_off_cr = cb_scalar_base + 11;

    const uint32_t unit_count = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_R_rm, cb_tilized);

    // Wait once on all 12 resident scalar tiles; never popped.
    for (uint32_t k = 0; k < 12; k++) {
        cb_wait_front(cb_scalar_base + k, 1);
    }

    for (uint32_t unit = 0; unit < unit_count; unit++) {
        // ---- Y ----
        for (uint32_t t = 0; t < y_tiles; t++) {
            tilize_one<cb_R_rm, cb_tilized>();
            mul_scalar_pack(cb_tilized, cb_wr_y, cb_partial);
            tilize_one<cb_G_rm, cb_tilized>();
            mul_scalar_pack(cb_tilized, cb_wg_y, cb_temp);
            add_and_pack(cb_partial, cb_temp, cb_partial);
            tilize_one<cb_B_rm, cb_tilized>();
            mul_scalar_pack(cb_tilized, cb_wb_y, cb_temp);
            add_and_pack(cb_partial, cb_temp, cb_partial);

            offset_clamp_pack(cb_partial, cb_off_y, cb_sum);
            compute_kernel_lib::untilize<
                1,
                cb_sum,
                cb_out_bf16,
                uc::InitUninitMode::InitAndUninit,
                uc::WaitMode::WaitBlock,
                uc::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
            typecast_and_pack(cb_out_bf16, cb_out_rm);
        }

        // ---- Cb then Cr: sum of 4 2x2 corners per channel (weights pre-scaled) ----
        for (uint32_t plane = 0; plane < 2; plane++) {
            const uint32_t cb_wr = plane == 0 ? cb_wr_cb : cb_wr_cr;
            const uint32_t cb_wg = plane == 0 ? cb_wg_cb : cb_wg_cr;
            const uint32_t cb_wb = plane == 0 ? cb_wb_cb : cb_wb_cr;
            const uint32_t cb_off = plane == 0 ? cb_off_cb : cb_off_cr;

            for (uint32_t t = 0; t < uv_tiles; t++) {
                tilize_one<cb_R_rm, cb_tilized>();
                mul_scalar_pack(cb_tilized, cb_wr, cb_partial);
                for (uint32_t c = 1; c < 4; c++) {
                    tilize_one<cb_R_rm, cb_tilized>();
                    mul_scalar_pack(cb_tilized, cb_wr, cb_temp);
                    add_and_pack(cb_partial, cb_temp, cb_partial);
                }
                for (uint32_t c = 0; c < 4; c++) {
                    tilize_one<cb_G_rm, cb_tilized>();
                    mul_scalar_pack(cb_tilized, cb_wg, cb_temp);
                    add_and_pack(cb_partial, cb_temp, cb_partial);
                }
                for (uint32_t c = 0; c < 4; c++) {
                    tilize_one<cb_B_rm, cb_tilized>();
                    mul_scalar_pack(cb_tilized, cb_wb, cb_temp);
                    add_and_pack(cb_partial, cb_temp, cb_partial);
                }

                offset_clamp_pack(cb_partial, cb_off, cb_sum);
                compute_kernel_lib::untilize<
                    1,
                    cb_sum,
                    cb_out_bf16,
                    uc::InitUninitMode::InitAndUninit,
                    uc::WaitMode::WaitBlock,
                    uc::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
                typecast_and_pack(cb_out_bf16, cb_out_rm);
            }
        }
    }
}
