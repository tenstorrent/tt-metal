// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Each input tile (bf16) -> DEST -> rounded onto the bfloat4_b grid (ckv_round_bfp4.hpp) -> packed to cb_out
// (bfloat4_b; the pack only copies bits) and, when CKV_ROUNDED_OUT, to cb_rounded (bf16: the same values).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ckv_round_bfp4.hpp"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
#ifdef CKV_ROUNDED_OUT
    constexpr uint32_t cb_rounded = get_compile_time_arg_val(2);
#endif

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_tile_init(cb_in);
    MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
    for (uint32_t t = 0; t < n_tiles; t++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
#ifndef CKV_NO_ROUND
        MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ckv_round_bfp4_face, 0, VectorMode::RC)));
#endif
        tile_regs_commit();
        cb_pop_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
#ifdef CKV_ROUNDED_OUT
        cb_reserve_back(cb_rounded, 1);
        pack_reconfig_data_format(cb_rounded);
        pack_tile(0, cb_rounded);
        cb_push_back(cb_rounded, 1);
#endif
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
