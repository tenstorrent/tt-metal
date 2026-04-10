// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford H-dimension reduction kernel.
// Reduces along H (rows) directly using the Welford's LLK, which natively reduces rows
// and maintains per-column accumulators.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    // Runtime arg: number of independent column-reductions this core must perform.
    // Each column-reduction processes Ht tiles vertically and produces one output tile.
    uint32_t NCWt = get_arg_val<uint32_t>(0);

    // Compile-time args:
    // Number of tiles along the H (reduction) dimension.
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    // The actual number of elements along H (before padding).
    constexpr uint32_t H = get_compile_time_arg_val(1);
    // Number of elements per tile in the H dimension (typically 32).
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    // Whether input scaling is required.
    constexpr bool do_scale = get_compile_time_arg_val(3) != 0;
    // Whether to apply Bessel's correction (divide by N-1 instead of N).
    constexpr bool correction = get_compile_time_arg_val(4) != 0;
    // Whether to compute standard deviation (sqrt of variance) instead of variance.
    constexpr bool is_std = get_compile_time_arg_val(5) != 0;

    constexpr uint32_t onetile = 1;

    // Circular buffer that the reader kernel fills with input tiles.
    constexpr auto cb_in = tt::CBIndex::c_0;
    // Scalar tile produced by the reader via generate_reduce_scaler.
    constexpr auto cb_scalar = tt::CBIndex::c_2;
    // Circular buffer where the final variance/std output tile is written.
    constexpr auto cb_out = tt::CBIndex::c_16;

    experimental::CircularBuffer cb_in_obj(cb_in);
    experimental::CircularBuffer cb_scalar_obj(cb_scalar);
    experimental::CircularBuffer cb_out_obj(cb_out);

    // Destination register indices inside the Tensix DST register file.
    // Welford's LLK uses three adjacent dst registers:
    //   input_dst (0) – scratch for the current input tile,
    //   mean_dst  (1) – running / final mean accumulator,
    //   var_dst   (2) – running / final variance accumulator.
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // The number of valid rows in the last tile in height dimension.
    // Welford's LLK processes rows naturally, so we skip padding rows
    // in the last tile via welford_update_rows.
    constexpr uint32_t last_tile_rows = ((H % tile_height) == 0) ? tile_height : (H % tile_height);

    compute_kernel_hw_startup(cb_in, cb_out);
    pack_reconfig_data_format(cb_out);

    if constexpr (do_scale) {
        // Scalar tile stays resident across all columns
        cb_scalar_obj.wait_front(onetile);
    }

    for (uint32_t ncwt = 0; ncwt < NCWt; ncwt++) {
        // Welford accumulation along the H dimension for one column of tiles.

        // start_N is the cumulative row count across tiles processed so far; passed
        // to the Welford LLK so it can compute the correct 1/(N+1) reciprocal
        // for each row's running-mean update.
        uint32_t start_N = 0;

        // Programs SFPU replay buffer + clears LREG4/5
        welford_init();

        // When scaling, DST must be acquired/released per tile because
        // the FPU mul and SFPU welford use incompatible pipeline configs
        // and the reverse transition (SFPU->FPU) requires a full DST
        // cycle to reset pipeline state.
        // Without scaling, copy_tile and welford (both SFPU-compatible)
        // can share a single DST window for the entire loop.
        if constexpr (!do_scale) {
            copy_tile_to_dst_init_short(cb_in);
            tile_regs_acquire();
        }

        // Welford SFPU state (running mean in LREG4, M2 in LREG5)
        // persists across DST cycles because LREGs are separate from
        // the DST register file managed by tile_regs_acquire/release.
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            cb_in_obj.wait_front(onetile);

            if constexpr (do_scale) {
                tile_regs_acquire();
                mul_tiles_bcast_scalar_init_short(cb_in, cb_scalar);
                mul_tiles_bcast_scalar(cb_in, cb_scalar, 0, 0, input_dst);
                // Reconfigure pipeline from FPU back to SFPU for welford.
                // While it's confusing to use copy_tile_to_dst_init_short here,
                // since no copy is done afterwards, there is no welford_init_short
                // function or some other generic way to reset the pipeline state.
                copy_tile_to_dst_init_short(cb_in);
            } else {
                copy_tile(cb_in, 0, input_dst);
            }
            cb_in_obj.pop_front(onetile);

            if (ht < (Ht - 1)) {
                welford_update<0>(input_dst, start_N, {});
                if constexpr (do_scale) {
                    // No-op PACK cycle: resets pipeline state for the
                    // next iteration's FPU mul_tiles_bcast_scalar.
                    tile_regs_commit();
                    tile_regs_wait();
                    tile_regs_release();
                }
            } else {
                // Last tile: process only valid rows, then finalize
                welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                // scale_idx controls the divisor for M2 -> variance conversion:
                //   correction=false: scale_idx = H-1, reciprocal = 1/H  (population variance)
                //   correction=true:  scale_idx = H-2, reciprocal = 1/(H-1) (sample variance)
                constexpr uint32_t scale_idx = correction ? (H - 2) : (H - 1);
                welford_finalize_to_row<0>(mean_dst, scale_idx, {});
                if constexpr (is_std) {
                    sqrt_tile_init();
                    sqrt_tile(var_dst);
                }
                tile_regs_commit();
            }
            start_N += tile_height;
        }

        // Pack variance/std directly to output -- no transpose needed for H reduction
        // because Welford natively produces results in row orientation which matches
        // the desired output layout (one row of results per column of input).
        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(var_dst, cb_out);
        tile_regs_release();
        cb_out_obj.push_back(onetile);
    }
}
