// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// rotary_embedding_indexed chain mode compute (deepseek_v41_flash_d_p P.22, ROPE_CHAINS). Per block (one head's
// 32 rows x W):
//   IN_RM: tilize the 32 row pages of cb_in into the WT tiles of cb_x (exact for bf16).
//   RoPE of the RT tiles at tile offset RO, the arithmetic of rotary_embedding_llama's compute kernel step for step
//     (rotated = x @ trans -> bf16; rotated * sin -> bf16; x * cos -> bf16; sum -> bf16), so the rotated tiles are
//     the stock op's bit for bit.
//   Assemble the WT output tiles in order (the pass-through tiles of x, the rotated ones at RO) into cb_asm in its
//     format: copy to DEST and pack (bf16; bfloat8_b = ttnn.typecast's pack of the same bf16 values; MBITS 3 / 7:
//     the bfp_round fork's SFPU round to nearest even onto the bfp grid first, then the pack only copies bits).
//   OUT_RM: untilize cb_asm (bf16) into cb_out (32 rows).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "chain_share.hpp"
#include "ttnn/ttnn/bringup/ckv_bfp4_ttnn/kernels/ckv_round_bfp4.hpp"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(3);
    constexpr uint32_t cb_trans = get_compile_time_arg_val(4);
    constexpr uint32_t cb_rot = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sin_i = get_compile_time_arg_val(6);
    constexpr uint32_t cb_cos_i = get_compile_time_arg_val(7);
    constexpr uint32_t cb_r = get_compile_time_arg_val(8);
    constexpr uint32_t cb_asm = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(10);
    constexpr uint32_t IN_RM = get_compile_time_arg_val(11);
    constexpr uint32_t OUT_RM = get_compile_time_arg_val(12);
    constexpr uint32_t WT = get_compile_time_arg_val(13);
    constexpr uint32_t RT = get_compile_time_arg_val(14);
    constexpr uint32_t RO = get_compile_time_arg_val(15);
    constexpr uint32_t GX = get_compile_time_arg_val(16);
    constexpr uint32_t MBITS = get_compile_time_arg_val(17);  // 0: no round (bf16 / bfloat8_b of bf16 values)
    constexpr uint32_t DST_TILES = get_compile_time_arg_val(18);

    uint32_t first, count;
    chain_share(get_common_arg_val<uint32_t>(0), get_common_arg_val<uint32_t>(1), GX, first, count);
    if (count == 0) {
        return;
    }

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_trans, cb_rot);
    cb_wait_front(cb_trans, 1);

    for (uint32_t k = 0; k < count; ++k) {
        if constexpr (IN_RM) {
            compute_kernel_lib::tilize<WT, cb_in, cb_x>(1, 32);
        }
        cb_wait_front(cb_x, WT);
        cb_wait_front(cb_cos, RT);
        cb_wait_front(cb_sin, RT);

        // rotated = x @ trans_mat (trans in1 / srcA, x in0 / srcB)
        cb_reserve_back(cb_rot, RT);
        reconfig_data_format<SrcOrder::Reverse>(cb_x, cb_trans);
        pack_reconfig_data_format(cb_rot);
        matmul_init(cb_x, cb_trans);
        tile_regs_acquire();
        for (uint32_t j = 0; j < RT; ++j) {
            matmul_tiles(cb_x, cb_trans, RO + j, 0, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < RT; ++j) {
            pack_tile(j, cb_rot);
        }
        tile_regs_release();
        cb_push_back(cb_rot, RT);

        // sin_interm = rotated * sin
        cb_wait_front(cb_rot, RT);
        cb_reserve_back(cb_sin_i, RT);
        reconfig_data_format(cb_rot, cb_sin);
        mul_init(cb_rot, cb_sin);
        tile_regs_acquire();
        for (uint32_t j = 0; j < RT; ++j) {
            mul_tiles(cb_rot, cb_sin, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < RT; ++j) {
            pack_tile(j, cb_sin_i);
        }
        tile_regs_release();
        cb_push_back(cb_sin_i, RT);
        cb_pop_front(cb_rot, RT);

        // cos_interm = x * cos
        cb_reserve_back(cb_cos_i, RT);
        reconfig_data_format(cb_x, cb_cos);
        mul_init(cb_x, cb_cos);
        tile_regs_acquire();
        for (uint32_t j = 0; j < RT; ++j) {
            mul_tiles(cb_x, cb_cos, RO + j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < RT; ++j) {
            pack_tile(j, cb_cos_i);
        }
        tile_regs_release();
        cb_push_back(cb_cos_i, RT);
        cb_pop_front(cb_cos, RT);
        cb_pop_front(cb_sin, RT);

        // r = cos_interm + sin_interm
        cb_wait_front(cb_cos_i, RT);
        cb_wait_front(cb_sin_i, RT);
        cb_reserve_back(cb_r, RT);
        reconfig_data_format(cb_cos_i, cb_sin_i);
        add_init(cb_cos_i, cb_sin_i);
        tile_regs_acquire();
        for (uint32_t j = 0; j < RT; ++j) {
            add_tiles(cb_cos_i, cb_sin_i, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < RT; ++j) {
            pack_tile(j, cb_r);
        }
        tile_regs_release();
        cb_push_back(cb_r, RT);
        cb_pop_front(cb_cos_i, RT);
        cb_pop_front(cb_sin_i, RT);

        // assemble the head's WT tiles in order into cb_asm (its format)
        cb_wait_front(cb_r, RT);
        cb_reserve_back(cb_asm, WT);
        reconfig_data_format_srca(cb_x);
        pack_reconfig_data_format(cb_asm);
        copy_init(cb_x);
        if constexpr (MBITS != 0) {
            MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
        }
        for (uint32_t t0 = 0; t0 < WT; t0 += DST_TILES) {
            const uint32_t nb = WT - t0 < DST_TILES ? WT - t0 : DST_TILES;
            tile_regs_acquire();
            for (uint32_t i = 0; i < nb; ++i) {
                const uint32_t t = t0 + i;
                if (t >= RO && t < RO + RT) {
                    copy_tile(cb_r, t - RO, i);
                } else {
                    copy_tile(cb_x, t, i);
                }
                if constexpr (MBITS != 0) {
                    MATH((_llk_math_eltwise_unary_sfpu_params_(
                        ckernel::sfpu::ckv_round_bfp_face<MBITS>, i, VectorMode::RC)));
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < nb; ++i) {
                pack_tile(i, cb_asm);
            }
            tile_regs_release();
        }
        cb_push_back(cb_asm, WT);
        cb_pop_front(cb_x, WT);
        cb_pop_front(cb_r, RT);

        if constexpr (OUT_RM) {
            compute_kernel_lib::untilize<WT, cb_asm, cb_out_rm>(1);
        }
    }
    cb_pop_front(cb_trans, 1);
}
