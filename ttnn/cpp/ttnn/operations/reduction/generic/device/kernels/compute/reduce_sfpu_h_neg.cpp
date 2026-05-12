// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU sibling of reduce_h_neg.cpp: MIN along H as -MAX(-x).
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
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
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
    // Chunk one less than DEST_AUTO_LIMIT so the binary max fold has a spare DST register
    // for its copy_tile destination (FPU folds in place and uses the full DEST_AUTO_LIMIT).
    constexpr uint32_t row_chunk = compute_kernel_lib::DEST_AUTO_LIMIT - 1;

    // Circular buffers:
    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_output = tt::CBIndex::c_3;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t work_dst = row_chunk;

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

    // H-axis MIN as -MAX(-x), chunked over `row_chunk` output columns at a time.
    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t wt_base = 0; wt_base < Wt; wt_base += row_chunk) {
            const uint32_t chunk_end = (wt_base + row_chunk < Wt) ? (wt_base + row_chunk) : Wt;
            const uint32_t current_chunk = chunk_end - wt_base;

            tile_regs_acquire();

            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t k = 0; k < current_chunk; ++k) {
                    cb_wait_front(cb_input, onetile);
                    if (ht == 0) {
                        copy_tile(cb_input, 0, k);
                        negative_tile_init();
                        negate(k);
                    } else {
                        copy_tile(cb_input, 0, work_dst);
                        negative_tile_init();
                        negate(work_dst);
                        compute_kernel_lib::detail::sfpu_reduce_max_fold_init<REDUCE_FORMAT>();
                        compute_kernel_lib::detail::sfpu_reduce_max_fold_tile<REDUCE_FORMAT>(k, work_dst, k);
                    }
                    cb_pop_front(cb_input, onetile);
                }
            }

            sfpu_reduce_init<REDUCE_OP, REDUCE_FORMAT>();
            for (uint32_t k = 0; k < current_chunk; ++k) {
                sfpu_reduce<REDUCE_OP, REDUCE_FORMAT, REDUCE_DIM>(k, /*ct_dim=*/1, /*rt_dim=*/1);
            }

            negative_tile_init();
            for (uint32_t k = 0; k < current_chunk; ++k) {
                negate(k);
            }

#ifdef REDUCE_POST_MUL
            for (uint32_t k = 0; k < current_chunk; ++k) {
                compute_kernel_lib::detail::sfpu_post_mul_tile<REDUCE_FORMAT>(k, post_mul_scaler_bits);
            }
#endif

            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < current_chunk; ++k) {
                cb_reserve_back(cb_output, onetile);
                pack_tile(k, cb_output);
                cb_push_back(cb_output, onetile);
            }
            tile_regs_release();
        }
    }

    PACK((llk_pack_reduce_mask_clear()));
}
