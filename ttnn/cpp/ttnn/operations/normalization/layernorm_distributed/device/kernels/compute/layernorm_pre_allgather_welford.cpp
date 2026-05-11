// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes layernorm statistics.
 * For layernorm it computes E(x**2) and E(x) and returns them as a two tile wide output tensor containing E(x**2) and
 * E(x) in the left most columns per tile. For rmsnorm it computes E(x**2) and returns it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/welford.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose_wh_dest.h"

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    namespace kutil = norm::kernel_util;
    namespace generic = kutil::generic;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
#if FUSE_PRE_ADD
    constexpr uint32_t blk = get_compile_time_arg_val(2);
#endif

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_14;
    constexpr uint32_t cb_x2 = tt::CBIndex::c_1;           // x**2
    constexpr uint32_t cb_reciprocals = tt::CBIndex::c_2;  // recip table
#if FUSE_PRE_ADD
    constexpr uint32_t cb_res = tt::CBIndex::c_5;         // residual b
    constexpr uint32_t cb_inp = tt::CBIndex::c_3;         // fused a + b (sized to a few blocks)
    constexpr uint32_t cb_mean_spill = tt::CBIndex::c_4;  // Welford mean accumulator spill (1 tile)
    constexpr uint32_t cb_m2_spill = tt::CBIndex::c_6;    // Welford M2 accumulator spill (1 tile)
#else
    constexpr uint32_t cb_inp = cb_in0;
#endif

#if FUSE_PRE_ADD
    binary_op_init_common(cb_in0, cb_res, cb_inp);
#else
    compute_kernel_hw_startup(cb_inp, cb_inp, cb_x2);
#endif
    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0);
    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    constexpr uint32_t last_tile_rows = (W % 32) == 0 ? 32 : W % 32;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr uint32_t dst0 = 0;
        constexpr uint32_t dst1 = 1;
        constexpr uint32_t dst2 = 2;

#if FUSE_PRE_ADD
        // Block-interleaved pre-add + Welford. The Welford accumulator lives in the SFPU within a
        // tile_regs scope, but the pre-add must use its own tile_regs scope to pack its result to
        // cb_inp before the Welford pass can transpose-read it back. To bridge those scopes the
        // accumulator (mean, M2) is spilled to cb_mean_spill / cb_m2_spill between chunks via
        // welford_save_state / welford_restore_state. This lets cb_inp stay sized to a small
        // number of tiles (blk * 2 for double-buffer) regardless of Wt. Larger blk amortizes the
        // save/restore overhead and accuracy loss across more tiles per spill cycle; blk is
        // chosen by the factory as gcd(Wt, DST capacity) so it always divides Wt.

        // Seed the spill CBs with an initialized (zero) Welford state,
        // since iteration 0 below expects it.
        tile_regs_acquire();
        welford_init();
        welford_save_state(dst1);
        tile_regs_commit();
        cb_reserve_back(cb_mean_spill, 1);
        cb_reserve_back(cb_m2_spill, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_mean_spill);
        pack_tile(dst1, cb_mean_spill);
        pack_tile(dst2, cb_m2_spill);
        tile_regs_release();
        cb_push_back(cb_mean_spill, 1);
        cb_push_back(cb_m2_spill, 1);

        uint32_t start_N = 0;
        for (auto block : generic::blocks(Wt, blk)) {
            // --- Pre-add: cb_in0 + cb_res -> cb_inp (block tiles in one tile_regs scope) ---
            reconfig_data_format(cb_in0, cb_res);
            pack_reconfig_data_format(cb_inp);
            add_tiles_init(cb_in0, cb_res);
            cb_wait_front(cb_in0, block.size());
            cb_wait_front(cb_res, block.size());
            cb_reserve_back(cb_inp, block.size());
            tile_regs_acquire();
            for (auto i : block.local()) {
                add_tiles(cb_in0, cb_res, i, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (auto i : block.local()) {
                pack_tile(i, cb_inp);
            }
            cb_push_back(cb_inp, block.size());
            tile_regs_release();
            cb_pop_front(cb_in0, block.size());
            cb_pop_front(cb_res, block.size());

            // --- Welford: reload accumulator, update with block tiles, spill back ---
            cb_wait_front(cb_mean_spill, 1);
            cb_wait_front(cb_m2_spill, 1);
            cb_wait_front(cb_inp, block.size());
            tile_regs_acquire();
            reconfig_data_format_srca(cb_in0, cb_mean_spill);
            copy_tile_init(cb_mean_spill);
            copy_tile(cb_mean_spill, 0, dst1);
            copy_tile_to_dst_init_short_with_dt(cb_mean_spill, cb_m2_spill);
            copy_tile(cb_m2_spill, 0, dst2);
            welford_restore_state(dst1);

            reconfig_data_format_srca(cb_m2_spill, cb_inp);
            transpose_wh_init_short(cb_inp);
            for (auto i : block.local()) {
                transpose_wh_tile(cb_inp, i, dst0);
                if (block.to_global(i) < Wt - 1) {
                    welford_update<W>(dst0, start_N, *p_reciprocals);
                } else {
                    welford_update_rows<W>(dst0, start_N, 0, last_tile_rows, *p_reciprocals);
                }
                start_N += 32;
            }
            welford_save_state(dst1);
            tile_regs_commit();
            cb_pop_front(cb_mean_spill, 1);
            cb_pop_front(cb_m2_spill, 1);
            cb_pop_front(cb_inp, block.size());
            cb_reserve_back(cb_mean_spill, 1);
            cb_reserve_back(cb_m2_spill, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_inp, cb_mean_spill);
            pack_tile(dst1, cb_mean_spill);
            pack_tile(dst2, cb_m2_spill);
            tile_regs_release();
            cb_push_back(cb_mean_spill, 1);
            cb_push_back(cb_m2_spill, 1);
        }

        // Finalize: reload accumulator and write mean and variance to cb_x2.
        cb_wait_front(cb_mean_spill, 1);
        cb_wait_front(cb_m2_spill, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_inp, cb_mean_spill);
        copy_tile_init(cb_mean_spill);
        copy_tile(cb_mean_spill, 0, dst1);
        copy_tile_to_dst_init_short_with_dt(cb_mean_spill, cb_m2_spill);
        copy_tile(cb_m2_spill, 0, dst2);
        welford_restore_state(dst1);
        welford_finalize_to_row<W>(dst1, W - 1, *p_reciprocals);
        tile_regs_commit();
        cb_pop_front(cb_mean_spill, 1);
        cb_pop_front(cb_m2_spill, 1);

        cb_reserve_back(cb_x2, 2);
        tile_regs_wait();
        pack_reconfig_data_format(cb_mean_spill, cb_x2);
        pack_tile(dst1, cb_x2);
        pack_tile(dst2, cb_x2);
        cb_push_back(cb_x2, 2);
        tile_regs_release();
#else
        reconfig_data_format(cb_inp, cb_inp);
        pack_reconfig_data_format(cb_x2);

        tile_regs_acquire();
        uint32_t start_N = 0;
        transpose_wh_init(cb_inp, cb_x2);
        welford_init();

        for (uint32_t wt = 0; wt < (Wt - 1); wt++) {
            cb_wait_front(cb_inp, 1);  // cumulative wait
            transpose_wh_tile(cb_inp, 0, dst0);
            // welford_tile<dst0, dst1, dst2, true, 0>((wt) * 32, W, 0, {});
            welford_update<W>(dst0, start_N, *p_reciprocals);
            start_N += 32;
            cb_pop_front(cb_inp, 1);
        }
        cb_wait_front(cb_inp, 1);  // cumulative wait
        transpose_wh_tile(cb_inp, 0, dst0);
        welford_update_rows<W>(dst0, start_N, 0, last_tile_rows, *p_reciprocals);
        cb_pop_front(cb_inp, 1);
        welford_finalize_to_row<W>(dst1, W - 1, *p_reciprocals);
        // tt-llk/issues/549
        // BUG: using transpose_dest here causes a bug. where the kernel hangs
        //  transpose_wh_dest_init_short();
        //  transpose_wh_dest(dst1);
        //  transpose_wh_dest(dst2);
        cb_reserve_back(cb_x2, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst1, cb_x2);
        pack_tile(dst2, cb_x2);
        cb_push_back(cb_x2, 2);
        tile_regs_release();
#endif

        reconfig_data_format(cb_x2, cb_x2);
        pack_reconfig_data_format(cb_out);
        transpose_wh_init_short(cb_x2);
        tile_regs_acquire();
        cb_wait_front(cb_x2, 2);  // cumulative wait
        transpose_wh_tile(cb_x2, 0, dst0);
        transpose_wh_tile(cb_x2, 1, dst1);
        cb_pop_front(cb_x2, 2);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_out);
        pack_tile(dst1, cb_out);
        cb_push_back(cb_out, 2);
        tile_regs_release();
    }
}
