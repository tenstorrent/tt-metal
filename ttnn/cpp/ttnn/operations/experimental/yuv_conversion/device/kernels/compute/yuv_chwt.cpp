// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: CHWT bfloat16 → YUV uint8 (degenerate-tile approach).
//
// CB pages are single T-chunks (32 bf16 = 64 bytes input/scratch,
// 32 uint8 = 32 bytes output). pack_tile writes exactly one CB page.
//
// Three passes (Y, Cb, Cr). For each (R,G,B) chunk triplet:
//   out = w_R*R + w_G*G + w_B*B + offset  →  clamp(0,255)  →  uint8
//
// Uses scratch CBs c_s0, c_s1 for intermediate scaled values.
// Every tile_regs_acquire MUST be paired with tile_regs_commit +
// tile_regs_wait + pack_tile + tile_regs_release.  Never call
// tile_regs_release without the commit/wait/pack sequence — that
// leaves PACK waiting for a commit that never arrives, causing a hang.
//
// Compile-time args:
//   [0] cb_R, [1] cb_G, [2] cb_B, [3] cb_s0, [4] cb_s1, [5] cb_out
//   [6] y_triplets, [7] uv_triplets
// Runtime args (float32 bits):
//   [0..3]  Y  : w_r, w_g, w_b, offset
//   [4..7]  Cb : w_r, w_g, w_b, offset
//   [8..11] Cr : w_r, w_g, w_b, offset

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

static constexpr uint32_t FMT_BF16 = 5u;
static constexpr uint32_t FMT_UINT8 = 1u;
// 255.0 as float32 bits
static constexpr uint32_t F32_255 = 0x437F0000u;

void kernel_main() {
    constexpr uint32_t cb_R = get_compile_time_arg_val(0);
    constexpr uint32_t cb_G = get_compile_time_arg_val(1);
    constexpr uint32_t cb_B = get_compile_time_arg_val(2);
    constexpr uint32_t cb_s0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_s1 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t y_triplets = get_compile_time_arg_val(6);
    constexpr uint32_t uv_triplets = get_compile_time_arg_val(7);

    const uint32_t w_r[3] = {get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(8)};
    const uint32_t w_g[3] = {get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(9)};
    const uint32_t w_b[3] = {get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(10)};
    const uint32_t off[3] = {get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(7), get_arg_val<uint32_t>(11)};
    const uint32_t ntiles[3] = {y_triplets, uv_triplets, uv_triplets};

    for (uint32_t pass = 0; pass < 3; pass++) {
        const uint32_t wr = w_r[pass], wg = w_g[pass], wb = w_b[pass], offset = off[pass];

        for (uint32_t t = 0; t < ntiles[pass]; t++) {
            // Step 1: R * w_R → c_s0
            tile_regs_acquire();
            cb_wait_front(cb_R, 1);
            copy_tile_init(cb_R);
            copy_tile(cb_R, 0, 0);
            cb_pop_front(cb_R, 1);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, wr);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_s0, 1);
            pack_tile(0, cb_s0);
            cb_push_back(cb_s0, 1);
            tile_regs_release();

            // Step 2: G * w_G → c_s1
            tile_regs_acquire();
            cb_wait_front(cb_G, 1);
            copy_tile_init(cb_G);
            copy_tile(cb_G, 0, 0);
            cb_pop_front(cb_G, 1);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, wg);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_s1, 1);
            pack_tile(0, cb_s1);
            cb_push_back(cb_s1, 1);
            tile_regs_release();

            // Step 3: R*wR + G*wG → c_s0
            tile_regs_acquire();
            cb_wait_front(cb_s0, 1);
            cb_wait_front(cb_s1, 1);
            add_tiles_init(cb_s0, cb_s1);
            add_tiles(cb_s0, cb_s1, 0, 0, 0);
            cb_pop_front(cb_s0, 1);
            cb_pop_front(cb_s1, 1);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_s0, 1);
            pack_tile(0, cb_s0);
            cb_push_back(cb_s0, 1);
            tile_regs_release();

            // Step 4: B * w_B → c_s1
            tile_regs_acquire();
            cb_wait_front(cb_B, 1);
            copy_tile_init(cb_B);
            copy_tile(cb_B, 0, 0);
            cb_pop_front(cb_B, 1);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, wb);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_s1, 1);
            pack_tile(0, cb_s1);
            cb_push_back(cb_s1, 1);
            tile_regs_release();

            // Step 5+6: final sum + offset, clamp, typecast → c_out
            tile_regs_acquire();
            cb_wait_front(cb_s0, 1);
            cb_wait_front(cb_s1, 1);
            add_tiles_init(cb_s0, cb_s1);
            add_tiles(cb_s0, cb_s1, 0, 0, 0);
            cb_pop_front(cb_s0, 1);
            cb_pop_front(cb_s1, 1);

            binop_with_scalar_tile_init();
            add_unary_tile(0, offset);

            relu_tile_init();
            relu_tile(0);

            unary_min_tile_init();
            unary_min_tile(0, F32_255);

            typecast_tile_init<FMT_BF16, FMT_UINT8>();
            typecast_tile<FMT_BF16, FMT_UINT8>(0);

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
    }
}
