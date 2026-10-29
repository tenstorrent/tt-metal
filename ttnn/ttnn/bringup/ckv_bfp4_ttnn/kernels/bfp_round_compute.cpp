// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// bf16 tiles -> the same tiles on the bfloat4_b (CKV_MBITS 3) / bfloat8_b (7) grid, rounded to nearest even like the
// host packer (ckv_round_bfp4.hpp), packed as cb_out's bfp format. Blocks of up to 16 tiles in a full-sync DEST; the
// precise pack only copies bits (the values are on the grid).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "ckv_round_bfp4.hpp"
#include "ckv_share.hpp"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

void kernel_main() {
#ifdef CKV_COMMON_GX  // (option compact: common [per, extra])
    uint32_t t_first, n_tiles;
    ckv_share(get_common_arg_val<uint32_t>(0), get_common_arg_val<uint32_t>(1), CKV_COMMON_GX, t_first, n_tiles);
#else
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
#endif
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(cb_in, cb_out);
    MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
    copy_init(cb_in);
    pack_init(cb_out);
    for (uint32_t t0 = 0; t0 < n_tiles; t0 += BLOCK) {
        const uint32_t nb = n_tiles - t0 < BLOCK ? n_tiles - t0 : BLOCK;
        cb_wait_front(cb_in, nb);
        tile_regs_acquire();
        for (uint32_t t = 0; t < nb; ++t) {
            copy_tile(cb_in, t, t);
            MATH((
                _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ckv_round_bfp_face<CKV_MBITS>, t, VectorMode::RC)));
        }
        tile_regs_commit();
        cb_pop_front(cb_in, nb);
        cb_reserve_back(cb_out, nb);
        tile_regs_wait();
        for (uint32_t t = 0; t < nb; ++t) {
            pack_tile(t, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, nb);
    }
}
