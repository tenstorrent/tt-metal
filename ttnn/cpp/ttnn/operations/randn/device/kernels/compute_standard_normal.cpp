// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
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
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

constexpr uint32_t neg_two = 0xc0000000u;        // -2.0f
constexpr uint32_t two_pi = 0x40c90fdbu;         //  2pi
constexpr uint32_t flt_min = 0x00800000u;        //  FLT_MIN
constexpr uint32_t one_minus_eps = 0x3F7FFFFFu;  //  A float value that does not exceed 1.0f

// Box-Muller pair `pair` gives the standard normals of stream tiles 2 * pair (Z1) and 2 * pair + 1 (Z2) from the
// uniforms rand_keyed_tile draws for those two tiles, so a tile's value does not depend on which core emits it.
inline void generate_standard_normal_pair(
    DataflowBuffer dfb_dst, uint32_t seed, uint32_t pair, bool emit_z1, bool emit_z2) {
    tile_regs_acquire();

    // reg0,reg1 <- U1,U2
    rand_keyed_tile_init();
    rand_keyed_tile(0, seed, 2 * pair, flt_min, one_minus_eps);
    rand_keyed_tile(1, seed, 2 * pair + 1, flt_min, one_minus_eps);

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
    if (emit_z2) {
        mul_binary_tile(1, 2, 2);
    }

    if (emit_z1) {
        // reg3 <- cos(U2 * 2pi)
        cos_tile_init();
        cos_tile(3);

        // reg3 <- Z1 = sqrt(ln(U1) * -2) * cos(U2 * 2pi)
        mul_binary_tile_init();
        mul_binary_tile(0, 3, 3);
    }

    if (emit_z2) {
        // reg2 <- sin(U2 * 2pi)
        sin_tile_init();
        sin_tile(2);

        // reg1 <- Z2 = sqrt(ln(U1) * -2) * sin(U2 * 2pi)
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 1);
    }

    tile_regs_commit();
    tile_regs_wait();

    // One tile per reservation: a core starting on an odd tile emits a lone Z2 before its first full pair, so a
    // two-tile reservation would straddle the end of the two-entry buffer.
    pack_reconfig_data_format(dfb::dst);
    if (emit_z1) {
        // pack Z1(reg3)
        dfb_dst.reserve_back(1);
        pack_tile(3, dfb::dst);
        dfb_dst.push_back(1);
    }
    if (emit_z2) {
        // pack Z2(reg1)
        dfb_dst.reserve_back(1);
        pack_tile(1, dfb::dst);
        dfb_dst.push_back(1);
    }

    tile_regs_release();
}

void kernel_main() {
    // -------------------------------------------------------------------------
    // Box-Muller transform
    //
    // Generate standard normal tiles via the Box-Muller transform, fused into a
    // single kernel.
    //
    // For each output pair:
    //   U1, U2 <- Uniform(FLT_MIN, 1-eps)
    //   R     = sqrt(ln(U1) * -2)
    //   Theta = 2*pi * U2
    //   Z1 = R * cos(Theta),  Z2 = R * sin(Theta)
    //
    // Pairs are aligned to even stream tiles; a core whose range starts on an odd
    // tile or ends on an even one emits only its half of that pair.
    // -------------------------------------------------------------------------

    const uint32_t seed = get_arg(args::seed);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t end_id = start_id + num_tiles;

    DataflowBuffer dfb_dst(dfb::dst);

    compute_kernel_hw_startup(dfb::dst, dfb::dst);
    copy_init(dfb::dst);
    for (uint32_t tile = start_id; tile < end_id;) {
        const uint32_t pair = tile / 2;
        const uint32_t z1_tile = 2 * pair;
        generate_standard_normal_pair(dfb_dst, seed, pair, tile == z1_tile, z1_tile + 1 < end_id);
        tile = z1_tile + 2;
    }
}  // void kernel_main()
