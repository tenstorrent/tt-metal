// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/rand.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#include "compute_kernel_api/eltwise_unary/typecast.h"

namespace NAMESPACE {

void MAIN {
    // -------------------------------------------------------------------------
    // Box-Muller transform
    //
    // Generate standard normal tiles via the Box–Muller transform, fused into a
    // single kernel without storing intermediates to memory.
    //
    // For each output pair:
    //   U1, U2 <- Uniform(0,1)
    //   R     = sqrt(ln(U1) * -2)
    //   Theta = 2*pi * U2
    //   Z1 = R * cos(Theta),  Z2 = R * sin(Theta)
    //
    // Emits 2 tiles per iteration; if num_tiles is odd, emits only Z1 for the
    // final tile. Optional OUTPUT_DTYPE_BFLOAT16 casts before packing.
    // -------------------------------------------------------------------------

    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t neg_two = 0xc0000000u;    // -2.0f
    constexpr uint32_t two_pi = 0x40c90fdbu;     //  2pi
    constexpr uint32_t flt_min = 0x00800000u;    //  FLT_MIN
    constexpr uint32_t one_minus = 0x3F7FFFFFu;  //  A float value that does not exceed 1.0f

    const uint32_t seed = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t num_pairs = num_tiles >> 1;
    const uint32_t is_odd = num_tiles & 1;

    init_sfpu(dst_cb_id, dst_cb_id);

    rand_tile_init(seed);
    log_tile_init();
    sqrt_tile_init();
    sin_tile_init();
    cos_tile_init();
    mul_binary_tile_init();
    fill_tile_init();
    binop_with_scalar_tile_init();
    typecast_tile_init();

    pack_reconfig_data_format(dst_cb_id);

    for (uint32_t p = 0; p < num_pairs; p++) {
        cb_reserve_back(dst_cb_id, 2);

        tile_regs_acquire();

        // U1
        rand_tile(0, flt_min, one_minus);
        // U2
        rand_tile(1, flt_min, one_minus);

        // sqrt(ln(U1) * -2)
        log_tile(0);
        mul_unary_tile(0, neg_two);
        sqrt_tile(0);

        fill_tile_bitcast(2, two_pi);

        // U2 * 2pi
        mul_binary_tile(1, 2, 3);
        mul_binary_tile(1, 2, 1);

        // cos(U2 * 2pi)
        cos_tile(3);
        // sin(U2 * 2pi)
        sin_tile(1);

        // Z1 = sqrt(ln(U1) * -2) * cos(U2 * 2pi)
        mul_binary_tile(0, 3, 3);
        // Z2 = sqrt(ln(U1) * -2) * sin(U2 * 2pi)
        mul_binary_tile(0, 1, 1);

#ifdef OUTPUT_DTYPE_BFLOAT16
        typecast_tile<0, 5>(3);
        typecast_tile<0, 5>(1);
#endif

        tile_regs_commit();
        tile_regs_wait();

        // Z1
        pack_tile(3, dst_cb_id);
        // Z2
        pack_tile(1, dst_cb_id);

        tile_regs_release();
        cb_push_back(dst_cb_id, 2);
    }

    if (is_odd) {
        cb_reserve_back(dst_cb_id, 1);

        tile_regs_acquire();

        // U1
        rand_tile(0, flt_min, one_minus);
        // U2
        rand_tile(1, flt_min, one_minus);

        // sqrt(ln(U1) * -2)
        log_tile(0);
        mul_unary_tile(0, neg_two);
        sqrt_tile(0);

        // U2 * 2pi
        mul_unary_tile(1, two_pi);

        // cos(U2 * 2pi)
        cos_tile(1);

        // Z1 = sqrt(ln(U1) * -2) * cos(U2 * 2pi)
        mul_binary_tile(0, 1, 0);

#ifdef OUTPUT_DTYPE_BFLOAT16
        typecast_tile<0, 5>(0);
#endif

        tile_regs_commit();
        tile_regs_wait();

        // Z1
        pack_tile(0, dst_cb_id);

        tile_regs_release();
        cb_push_back(dst_cb_id, 1);
    }
}
}  // namespace NAMESPACE
