// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"

/**
 * Register-based argmax along a non-HW (N or C) dim for TILE-layout inputs.
 *
 * For each output tile, the reader pushes `num_reduce_tiles` value tiles into
 * cb_in0 (fp32 / bf16 input values) and a matching number of fp32 index tiles
 * into cb_in1 (tile k is filled with the scalar (float)k).
 *
 * The compute kernel tracks, in DST registers only (no intermediate spill to
 * L1), two running accumulators per output tile:
 *   slot 0: max_val (fp32)
 *   slot 1: argmax (fp32, stored as float but always holding an integer value)
 * and uses two scratch slots (2,3) to load new input / mask per step.
 *
 * The argmax update is done arithmetically to avoid the SFPU `where`
 * ternary op; for a boolean mask m in {0.0f, 1.0f}:
 *
 *   argmax += m * (new_idx - argmax)
 *
 * which selects `new_idx` when the new value is strictly greater than the
 * running max, and keeps `argmax` otherwise. At the end of the reduction
 * `argmax` is typecast from fp32 to uint32 before packing.
 */
void kernel_main() {
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_reduce_tiles = get_compile_time_arg_val(1);

    constexpr auto cb_val = tt::CBIndex::c_0;
    constexpr auto cb_idx = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t onetile = 1;

    // Slot layout inside DST (32-bit mode):
    //   slot 0: max_val       (holding max value)
    //   slot 1: argmax        (holding float(index))
    //   slot 2: new_val / new_idx (scratch)
    //   slot 3: mask / delta      (scratch)
    constexpr uint32_t dst_max = 0;
    constexpr uint32_t dst_argmax = 1;
    constexpr uint32_t dst_scratch_a = 2;
    constexpr uint32_t dst_scratch_b = 3;

    init_sfpu(cb_val, cb_out);
    // SFPU op inits: each sets persistent SFPU state; safe to call once here—none
    // of these inits overwrite state another op still needs.
    binary_max_tile_init();
    gt_binary_tile_init();
    add_binary_tile_init();
    sub_binary_tile_init();
    mul_binary_tile_init();
    typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::UInt32>();

    for (uint32_t out_i = 0; out_i < num_output_tiles; ++out_i) {
        tile_regs_acquire();

        // --- Initialize running accumulators from tile k=0 ---
        // Ensure the unpacker is configured for cb_val. On the first output
        // tile this is a no-op (init_sfpu already configured cb_val), but on
        // subsequent iterations the previous output tile left the unpacker
        // configured for cb_idx, which silently re-interprets bf16 values as
        // fp32 on the next copy_tile(cb_val, ...) and breaks the reduction.
        copy_tile_to_dst_init_short_with_dt(cb_idx, cb_val);

        // max_val <- val[0]
        cb_wait_front(cb_val, onetile);
        copy_tile(cb_val, 0, dst_max);
        cb_pop_front(cb_val, onetile);

        // argmax <- idx[0]   (format switch: val -> idx)
        copy_tile_to_dst_init_short_with_dt(cb_val, cb_idx);
        cb_wait_front(cb_idx, onetile);
        copy_tile(cb_idx, 0, dst_argmax);
        cb_pop_front(cb_idx, onetile);

        // --- Reduce over remaining tiles k=1 .. num_reduce_tiles-1 ---
        for (uint32_t k = 1; k < num_reduce_tiles; ++k) {
            // new_val -> scratch_a  (format switch: idx -> val)
            copy_tile_to_dst_init_short_with_dt(cb_idx, cb_val);
            cb_wait_front(cb_val, onetile);
            copy_tile(cb_val, 0, dst_scratch_a);
            cb_pop_front(cb_val, onetile);

            // mask (0.0f or 1.0f) = (new_val > max_val) -> scratch_b
            gt_binary_tile(dst_scratch_a, dst_max, dst_scratch_b);

            // max_val = max(max_val, new_val)
            binary_max_tile(dst_max, dst_scratch_a, dst_max);

            // Overwrite scratch_a with new_idx (format switch: val -> idx)
            copy_tile_to_dst_init_short_with_dt(cb_val, cb_idx);
            cb_wait_front(cb_idx, onetile);
            copy_tile(cb_idx, 0, dst_scratch_a);
            cb_pop_front(cb_idx, onetile);

            // scratch_a = new_idx - argmax
            sub_binary_tile(dst_scratch_a, dst_argmax, dst_scratch_a);

            // scratch_a = mask * (new_idx - argmax)
            mul_binary_tile(dst_scratch_a, dst_scratch_b, dst_scratch_a);

            // argmax += mask * (new_idx - argmax)
            add_binary_tile(dst_argmax, dst_scratch_a, dst_argmax);
        }

        // Convert argmax fp32 -> uint32 in-place in slot `dst_argmax`.
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::UInt32>(dst_argmax);

        tile_regs_commit();

        cb_reserve_back(cb_out, onetile);
        tile_regs_wait();
        pack_tile(dst_argmax, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, onetile);
    }
}
