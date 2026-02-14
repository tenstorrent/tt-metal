// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/trigonometry.h"

namespace NAMESPACE {

constexpr uint32_t neg_two = 0xc0000000u;        // -2.0f
constexpr uint32_t two_pi = 0x40c90fdbu;         //  2pi
constexpr uint32_t flt_min = 0x00800000u;        //  FLT_MIN
constexpr uint32_t one_minus_eps = 0x3F7FFFFFu;  //  A float value that does not exceed 1.0f

template <bool EmitZ2>
inline void generate_standard_normal_tiles(uint32_t dst_cb_id) {
    constexpr uint32_t num_out_tiles = EmitZ2 ? 2 : 1;

    cb_reserve_back(dst_cb_id, num_out_tiles);

    tile_regs_acquire();

    // // reg0,reg1 <- U1,U2
    rand_tile(0, flt_min, one_minus_eps);
    rand_tile(1, flt_min, one_minus_eps);

    // reg0 <- sqrt(ln(U1) * -2)
    log_tile_init();
    log_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, neg_two);
    sqrt_tile_init();
    sqrt_tile(0);

    // reg2 <- 2pi
    fill_tile_init();
    fill_tile_bitcast(2, two_pi);

    // reg3,reg2 <- U2 * 2pi
    mul_binary_tile_init();
    mul_binary_tile(1, 2, 3);
    mul_binary_tile(1, 2, 2);

    // reg3 <- cos(U2 * 2pi)
    cos_tile_init();
    cos_tile(3);

    // reg3 <- Z1 = sqrt(ln(U1) * -2) * cos(U2 * 2pi)
    mul_binary_tile_init();
    mul_binary_tile(0, 3, 3);

    if constexpr (EmitZ2) {
        // reg2 <- sin(U2 * 2pi)
        sin_tile_init();
        sin_tile(2);

        // reg1 <- Z2 = sqrt(ln(U1) * -2) * sin(U2 * 2pi)
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 1);
    }

    tile_regs_commit();
    tile_regs_wait();

    pack_reconfig_data_format(dst_cb_id);
    // pack Z1(reg3)
    pack_tile(3, dst_cb_id);
    if constexpr (EmitZ2) {
        // pack Z2(reg1)
        pack_tile(1, dst_cb_id);
    }

    tile_regs_release();

    cb_push_back(dst_cb_id, num_out_tiles);
}

void MAIN {
    // -------------------------------------------------------------------------
    // Box-Muller transform
    //
    // Generate standard normal tiles via the Box–Muller transform, fused into a
    // single kernel.
    //
    // For each output pair:
    //   U1, U2 <- Uniform(FLT_MIN, 1-eps)
    //   R     = sqrt(ln(U1) * -2)
    //   Theta = 2*pi * U2
    //   Z1 = R * cos(Theta),  Z2 = R * sin(Theta)
    //
    // Emits 2 tiles per iteration; if num_tiles is odd, emits only Z1 for the
    // final tile.
    // -------------------------------------------------------------------------

    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);
    const uint32_t seed = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t num_pairs = num_tiles >> 1;
    const uint32_t is_odd = num_tiles & 1;
    const uint32_t num_rand_tiles = num_tiles + is_odd;

    init_sfpu(dst_cb_id, dst_cb_id);
    rand_tile_init(seed);
    for (uint32_t p = 0; p < num_pairs; p++) {
        generate_standard_normal_tiles<true>(dst_cb_id);
    }
    if (is_odd) {
        generate_standard_normal_tiles<false>(dst_cb_id);
    }
}
}  // namespace NAMESPACE
