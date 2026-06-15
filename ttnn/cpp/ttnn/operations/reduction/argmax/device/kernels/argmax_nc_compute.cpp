// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_binary_sfpu.h"

/**
 * Register-based argmax along a non-HW (N or C) dim for TILE-layout inputs.
 *
 * For each output tile, the reader pushes `num_reduce_tiles` value tiles into
 * cb_in0 (fp32 / bf16). Indices are NOT staged through a CB — we materialize
 * them as uint32 scalars directly inside DST via fill_tile_int<UInt32>. This
 * matters because tt-metal's host-side unpack-to-dest plumbing only honors
 * UnpackToDestFp32 for Float32 CBs (see jit_build/data_format.cpp); a UInt32 CB
 * is silently forced through the SrcA datapath, which when alternated with the
 * bf16/fp32 value CB corrupts the low 16 bits of each uint32 index slot.
 *
 * The kernel tracks two running accumulators per output tile in DST:
 *   slot 0: max_val (fp value)
 *   slot 1: argmax  (int32 index, packed directly to UInt32 output)
 * plus two scratch slots (2,3) for the per-step new_val / new_idx / mask.
 *
 * Both the max-value and argmax updates run through `where_tile`, just with
 * different DataFormat template parameters (Float32 / Int32) — same SFPU op,
 * different load/store mod. Using a single `where_tile_init()` avoids an
 * SFPCONFIG macro collision: `binary_max_min_init` and `where_tile_init` both
 * write SFPCONFIG macros 0 and 1, so they can't coexist in the same kernel.
 *
 *   mask   = (new_val > max_val)                            // gt_binary_tile, raw 0/1
 *   max    = where<Float32>(mask, new_val, max)             // fp32 ternary
 *   argmax = where<Int32>  (mask, new_idx, argmax)          // int32 ternary
 *
 * `where_tile` checks the condition slot with SFPSETCC's LREG_EQ0 modifier —
 * any nonzero bit pattern is true, all-zero is false — so the raw output of
 * `gt_binary_tile` feeds it directly without conversion. Strict `>` keeps the
 * first matching index on ties, matching PyTorch.
 */
void kernel_main() {
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_reduce_tiles = get_compile_time_arg_val(1);

    constexpr auto cb_val = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t onetile = 1;

    // Slot layout inside DST (32-bit mode):
    //   slot 0: max_val       (fp value)
    //   slot 1: argmax        (uint32 index)
    //   slot 2: new_val / new_idx (scratch)
    //   slot 3: mask              (scratch, raw 0/1 from gt_binary_tile)
    constexpr uint32_t dst_max = 0;
    constexpr uint32_t dst_argmax = 1;
    constexpr uint32_t dst_scratch_a = 2;
    constexpr uint32_t dst_scratch_b = 3;

    init_sfpu(cb_val, cb_out);
    // SFPU op inits. `where_tile_init` and `binary_max_min_init` BOTH install
    // SFPCONFIG macros 0 and 1 with different semantics, so they cannot be
    // active simultaneously. We sidestep the collision by using `where_tile`
    // for BOTH the max update (over fp32 values) and the argmax update (over
    // int32 indices) — same op, only the DataFormat template parameter differs.
    gt_binary_tile_init();
    where_tile_init();
    fill_tile_init();

    for (uint32_t out_i = 0; out_i < num_output_tiles; ++out_i) {
        tile_regs_acquire();

        // --- Initialize running accumulators from tile k=0 ---
        // max_val <- val[0]
        cb_wait_front(cb_val, onetile);
        copy_tile(cb_val, 0, dst_max);
        cb_pop_front(cb_val, onetile);

        // argmax <- 0 (uint32, materialized in-place via SFPU)
        fill_tile_int<DataFormat::UInt32>(dst_argmax, 0u);

        // --- Reduce over remaining tiles k=1 .. num_reduce_tiles-1 ---
        for (uint32_t k = 1; k < num_reduce_tiles; ++k) {
            // new_val -> scratch_a
            cb_wait_front(cb_val, onetile);
            copy_tile(cb_val, 0, dst_scratch_a);
            cb_pop_front(cb_val, onetile);

            // mask (raw 0 / 1) = (new_val > max_val) -> scratch_b
            gt_binary_tile(dst_scratch_a, dst_max, dst_scratch_b);

            // max_val = mask ? new_val : max_val  (fp32 ternary; in-place)
            where_tile<DataFormat::Float32>(dst_scratch_b, dst_scratch_a, dst_max, dst_max);

            // Overwrite scratch_a with new_idx = k (uint32) directly via SFPU,
            // bypassing the CB / SrcA unpack path entirely.
            fill_tile_int<DataFormat::UInt32>(dst_scratch_a, k);

            // argmax = mask ? new_idx : argmax   (int32 ternary; in-place)
            where_tile<DataFormat::Int32>(dst_scratch_b, dst_scratch_a, dst_argmax, dst_argmax);
        }

        tile_regs_commit();

        cb_reserve_back(cb_out, onetile);
        tile_regs_wait();
        pack_tile(dst_argmax, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, onetile);
    }
}
