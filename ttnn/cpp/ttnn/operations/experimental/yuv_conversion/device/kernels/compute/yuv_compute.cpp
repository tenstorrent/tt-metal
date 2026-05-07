// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-based compute kernel for YUV conversion.
//
// Processes 32 spatial positions per tile (full tile utilization).
// Reader fills row-major bf16 CBs (32 sticks × 32 columns per tile).
// This kernel tilizes, does linear combination with scalar broadcast,
// clamps [0, 255], untilizes back to row-major bf16, then typecasts
// bf16→uint8 via SFPU, and pushes to output CB.
//
// Three sequential phases (Y, Cb, Cr), each with its own scalar coefficients.
// The scalar tile CBs are populated by the reader and popped here between phases.
//
// Y tile:  tilize(R)*wr + tilize(G)*wg + tilize(B)*wb + off → clamp → untilize → typecast
// UV tile: (sum of 4 corners per channel, each multiplied by weight) + off → clamp → untilize → typecast
//          Reader pre-scales UV weights by 0.25, so sum*w*0.25 = avg*w.
//
// Compile-time args:
//   [0]  cb_R_rm     [1]  cb_G_rm     [2]  cb_B_rm
//   [3]  cb_tilized  [4]  cb_partial  [5]  cb_temp
//   [6]  cb_wr       [7]  cb_wg       [8]  cb_wb       [9]  cb_off
//   [10] cb_sum      [11] cb_out_rm   [12] num_y_tiles
//   [13] num_uv_tiles_per_plane       [14] num_t_tiles
//   [15] cb_out_bf16

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

// SFPU typecast: bf16 → uint8 on a row-major page (32x32 elements).
// copy_tile/pack_tile are element-wise inverses, so row-major layout is preserved.
// Reconfigures unpack/pack via init_sfpu; the next tilize with
// UnpackAndPackReconfigure restores bf16 state.
//
// On Blackhole, pack_untilize_dest_init enables DEST remap (remap_addrs + swizzle_32b)
// which is NOT cleared by pack_untilize_uninit. The FPU (copy_tile, pack_tile) honours
// the remap but the SFPU does not, so we must disable it before running the SFPU.
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
    constexpr uint32_t cb_out_bf16 = get_compile_time_arg_val(15);

    compute_kernel_hw_startup(cb_R_rm, cb_tilized);

    // ---- Phase 1: Y pass ----
    cb_wait_front(cb_wr, 1);
    cb_wait_front(cb_wg, 1);
    cb_wait_front(cb_wb, 1);
    cb_wait_front(cb_off, 1);

    for (uint32_t t = 0; t < num_y_tiles; t++) {
        compute_kernel_lib::tilize<
            1,
            cb_R_rm,
            cb_tilized,
            tc::InitUninitMode::InitAndUninit,
            tc::WaitMode::WaitBlock,
            tc::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
        mul_scalar_pack(cb_tilized, cb_wr, cb_partial);

        compute_kernel_lib::tilize<
            1,
            cb_G_rm,
            cb_tilized,
            tc::InitUninitMode::InitAndUninit,
            tc::WaitMode::WaitBlock,
            tc::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
        mul_scalar_pack(cb_tilized, cb_wg, cb_temp);
        add_and_pack(cb_partial, cb_temp, cb_partial);

        compute_kernel_lib::tilize<
            1,
            cb_B_rm,
            cb_tilized,
            tc::InitUninitMode::InitAndUninit,
            tc::WaitMode::WaitBlock,
            tc::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
        mul_scalar_pack(cb_tilized, cb_wb, cb_temp);
        add_and_pack(cb_partial, cb_temp, cb_partial);

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

    cb_pop_front(cb_wr, 1);
    cb_pop_front(cb_wg, 1);
    cb_pop_front(cb_wb, 1);
    cb_pop_front(cb_off, 1);

    // ---- Phases 2 & 3: Cb and Cr passes ----
    for (uint32_t plane = 0; plane < 2; plane++) {
        cb_wait_front(cb_wr, 1);
        cb_wait_front(cb_wg, 1);
        cb_wait_front(cb_wb, 1);
        cb_wait_front(cb_off, 1);

        for (uint32_t t = 0; t < num_uv_tiles_per_plane; t++) {
            compute_kernel_lib::tilize<
                1,
                cb_R_rm,
                cb_tilized,
                tc::InitUninitMode::InitAndUninit,
                tc::WaitMode::WaitBlock,
                tc::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
            mul_scalar_pack(cb_tilized, cb_wr, cb_partial);

            for (uint32_t c = 1; c < 4; c++) {
                compute_kernel_lib::tilize<
                    1,
                    cb_R_rm,
                    cb_tilized,
                    tc::InitUninitMode::InitAndUninit,
                    tc::WaitMode::WaitBlock,
                    tc::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
                mul_scalar_pack(cb_tilized, cb_wr, cb_temp);
                add_and_pack(cb_partial, cb_temp, cb_partial);
            }

            for (uint32_t c = 0; c < 4; c++) {
                compute_kernel_lib::tilize<
                    1,
                    cb_G_rm,
                    cb_tilized,
                    tc::InitUninitMode::InitAndUninit,
                    tc::WaitMode::WaitBlock,
                    tc::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
                mul_scalar_pack(cb_tilized, cb_wg, cb_temp);
                add_and_pack(cb_partial, cb_temp, cb_partial);
            }

            for (uint32_t c = 0; c < 4; c++) {
                compute_kernel_lib::tilize<
                    1,
                    cb_B_rm,
                    cb_tilized,
                    tc::InitUninitMode::InitAndUninit,
                    tc::WaitMode::WaitBlock,
                    tc::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
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

        cb_pop_front(cb_wr, 1);
        cb_pop_front(cb_wg, 1);
        cb_pop_front(cb_wb, 1);
        cb_pop_front(cb_off, 1);
    }
}
