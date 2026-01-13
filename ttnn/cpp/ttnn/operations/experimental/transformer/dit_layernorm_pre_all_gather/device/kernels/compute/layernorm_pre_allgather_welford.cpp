// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * LayerNorm-only Welford pre-allgather.
 * Computes E(x**2) and E(x) per row and outputs two TILE columns (x2, x).
 */

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/welford.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "compute_kernel_api/compute_kernel_hw_startup.h"
#include "compute_kernel_api/transpose_wh_dest.h"

namespace NAMESPACE {
void MAIN {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    namespace kutil = norm::kernel_util;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_14;
    constexpr uint32_t cb_x2 = tt::CBIndex::c_1;           // x**2
    constexpr uint32_t cb_reciprocals = tt::CBIndex::c_2;  // recip table

    compute_kernel_hw_startup(cb_inp, cb_inp, cb_x2);
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0);
    constexpr uint32_t last_tile_rows = (W % 32) == 0 ? 32 : W % 32;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr uint32_t dst0 = 0;
        constexpr uint32_t dst1 = 1;
        constexpr uint32_t dst2 = 2;

        reconfig_data_format(cb_inp, cb_inp);
        pack_reconfig_data_format(cb_x2);

        tile_regs_acquire();
        uint32_t start_N = 0;
        transpose_wh_init(cb_inp, cb_x2);
        welford_init();

        for (uint32_t wt = 0; wt < Wt; wt += block_size) {
            cb_wait_front(cb_inp, block_size);
            uint32_t r;
            for (r = 0; r < block_size && wt + r < Wt - 1; r++) {
                transpose_wh_tile(cb_inp, r, dst0);
                welford_update<W>(dst0, start_N, *p_reciprocals);
                start_N += 32;
            }
            if (wt + r == Wt - 1) {
                // This block contains the last tile
                transpose_wh_tile(cb_inp, r, dst0);
                welford_update_rows<W>(dst0, start_N, 0, last_tile_rows, *p_reciprocals);
            }
            cb_pop_front(cb_inp, block_size);
        }

        welford_finalize_to_row<W>(dst1, W - 1, *p_reciprocals);

        cb_reserve_back(cb_x2, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst1, cb_x2);
        pack_tile(dst2, cb_x2);
        cb_push_back(cb_x2, 2);
        tile_regs_release();
        reconfig_data_format(cb_x2, cb_x2);
        pack_reconfig_data_format(cb_out);
        transpose_wh_init_short(cb_x2);
        tile_regs_acquire();
        cb_wait_front(cb_x2, 2);
        transpose_wh_tile(cb_x2, 0, dst0);
        transpose_wh_tile(cb_x2, 1, dst1);
        cb_pop_front(cb_x2, 2);
        cb_reserve_back(cb_out, 2);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_out);
        pack_tile(dst1, cb_out);
        cb_push_back(cb_out, 2);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
