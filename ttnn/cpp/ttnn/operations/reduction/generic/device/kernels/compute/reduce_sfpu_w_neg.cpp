// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU sibling of reduce_w_neg.cpp: MIN along W as -MAX(-x).
// Format-aware via REDUCE_FORMAT (Int32 or Float32). Negate is the only MIN-specific
// step and stays here; the MAX-fold and post-mul logic is shared with the non-neg path.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_sfpu_helpers_compute.hpp"

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

    // Format-specific negate (the only MIN-specific op in this kernel).
    auto negate = [](uint32_t dst) {
        if constexpr (REDUCE_FORMAT == DataFormat::Int32) {
            negative_tile_int32(dst);
        } else {
            negative_tile(dst);
        }
    };

    init_sfpu(cb_input, cb_output);
    copy_tile_to_dst_init_short(cb_input);

    cb_wait_front(cb_scaler, onetile);

    PACK((llk_pack_reduce_mask_config<false /*untilize*/, REDUCE_DIM>()));

    // W-axis reduce: outer iterates output rows (Ht), inner folds Wt input tiles per output.
    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            tile_regs_acquire();

            if (Wt > 1) {
                compute_kernel_lib::detail::sfpu_reduce_max_fold_init<REDUCE_FORMAT>();
            }

            cb_wait_front(cb_input, onetile);
            copy_tile(cb_input, 0, acc_dst);
            cb_pop_front(cb_input, onetile);
            negative_tile_init();
            negate(acc_dst);

            for (uint32_t wt = 1; wt < Wt; ++wt) {
                cb_wait_front(cb_input, onetile);
                copy_tile(cb_input, 0, work_dst);
                negative_tile_init();
                negate(work_dst);
                compute_kernel_lib::detail::sfpu_reduce_max_fold_init<REDUCE_FORMAT>();
                compute_kernel_lib::detail::sfpu_reduce_max_fold_tile<REDUCE_FORMAT>(acc_dst, work_dst, acc_dst);
                cb_pop_front(cb_input, onetile);
            }

            sfpu_reduce_init<REDUCE_OP, REDUCE_FORMAT>();
            sfpu_reduce<REDUCE_OP, REDUCE_FORMAT, REDUCE_DIM>(acc_dst, /*ct_dim=*/1, /*rt_dim=*/1);

            negative_tile_init();
            negate(acc_dst);

#ifdef REDUCE_POST_MUL
            compute_kernel_lib::detail::sfpu_post_mul_tile<REDUCE_FORMAT>(acc_dst, post_mul_scaler_bits);
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
