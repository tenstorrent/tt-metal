// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU sibling of reduce_h_neg.cpp: Int32 MIN along H as -MAX(-x).

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/typecast.h"
#endif

#ifdef TRISC_PACK
#include "llk_pack_api.h"
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
#ifdef REDUCE_POST_MUL
    constexpr uint32_t post_mul_scaler_bits = get_compile_time_arg_val(3);
#endif

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_output = tt::CBIndex::c_3;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t acc_dst = 0;
    constexpr uint32_t work_dst = 1;

    init_sfpu(cb_input, cb_output);
    copy_tile_to_dst_init_short(cb_input);

    cb_wait_front(cb_scaler, onetile);

    PACK((llk_pack_reduce_mask_config<false /*untilize*/, REDUCE_DIM>()));

    // H-axis reduce: outer iterates output columns (Wt), inner folds Ht input tiles per output.
    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            tile_regs_acquire();

            if (Ht > 1) {
                binary_max_int32_tile_init();
            }

            cb_wait_front(cb_input, onetile);
            copy_tile(cb_input, 0, acc_dst);
            cb_pop_front(cb_input, onetile);
            negative_tile_init();
            negative_tile_int32(acc_dst);

            for (uint32_t ht = 1; ht < Ht; ++ht) {
                cb_wait_front(cb_input, onetile);
                copy_tile(cb_input, 0, work_dst);
                negative_tile_init();
                negative_tile_int32(work_dst);
                binary_max_int32_tile_init();
                binary_max_int32_tile(acc_dst, work_dst, acc_dst);
                cb_pop_front(cb_input, onetile);
            }

            sfpu_reduce_init<REDUCE_OP, REDUCE_FORMAT>();
            sfpu_reduce<REDUCE_OP, REDUCE_FORMAT, REDUCE_DIM>(acc_dst, /*ct_dim=*/1, /*rt_dim=*/1);

            negative_tile_init();
            negative_tile_int32(acc_dst);

#ifdef REDUCE_POST_MUL
            // sfpu_reduce leaves Int32 bits in DST; mul_unary_tile is fp32-only.
            // Cast Int32 -> fp32, multiply, then cast fp32 -> Int32 (truncates toward zero).
            typecast_tile_init<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>();
            typecast_tile<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>(acc_dst);
            binop_with_scalar_tile_init();
            mul_unary_tile(acc_dst, post_mul_scaler_bits);
            typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>();
            typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>(acc_dst);
#endif

            tile_regs_commit();

            cb_reserve_back(cb_output, onetile);
            tile_regs_wait();
            pack_tile(acc_dst, cb_output);
            tile_regs_release();
            cb_push_back(cb_output, onetile);
        }
    }

    PACK((llk_pack_reduce_mask_clear()));
}
