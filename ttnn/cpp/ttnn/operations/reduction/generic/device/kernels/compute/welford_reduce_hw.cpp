// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford HW-dimension reduction kernel (compute side).
//
// Phase 1 (per output): For each of reduce_batch_size NC slices,
// H-reduces each of Wt columns using the Welford LLK, finalizes to
// row format (welford_finalize_to_row) and packs the mean+var tile
// pair to cb_partial for the writer kernel to W-combine using the
// parallel Welford merge formula.
//
// Phase 2 (per output): Reads the combined Float32 scalar tile from
// cb_combined (produced by the writer after W-combining all partials
// and applying Bessel's correction), applies sqrt_tile when computing
// std, applies the user scalar via SFPU post-multiplication, and
// re-packs to cb_out in the output data format.  This ensures
// the packer hardware handles format conversion (required for
// BFLOAT8_B and for matching the output dtype to the input dtype).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

#ifdef WELFORD_POST_MUL
// SFPU multiply-by-scalar (mul_unary_tile) applied to the reduced output. See issue #45222.
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    // Runtime arg: total number of NC slices this core must process.
    uint32_t NC_per_core = get_arg_val<uint32_t>(0);

    // Compile-time args:
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
#ifdef WELFORD_POST_MUL
    // Packed fp32 post-multiplier applied to the reduced output via mul_unary_tile (SFPU).
    // For var this is scalar^2, for std it is |scalar| (see welford_reduce_program_factory).
    constexpr uint32_t post_mul_scaler_bits = get_compile_time_arg_val(4);
#endif
    constexpr uint32_t reduce_batch_size = get_compile_time_arg_val(5);
    constexpr bool is_std = get_compile_time_arg_val(6) != 0;

    constexpr uint32_t onetile = 1;

    // cb_in: For FP32 input it is flagged UnpackToDestFp32 (program factory), preserving FP32
    // mantissa for the copy_tile -> welford SFPU consumer path. BF16 input: Default.
    constexpr auto cb_in = tt::CBIndex::c_0;
    // Final output CB (output data format), consumed by the writer for NOC write.
    constexpr auto cb_out = tt::CBIndex::c_16;
    // Intermediate CB for mean+var tile pairs, consumed by writer kernel.
    constexpr auto cb_partial = tt::CBIndex::c_21;
    // Combined scalar result from the writer kernel (Float32).
    constexpr auto cb_combined = tt::CBIndex::c_22;

    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_out_obj(cb_out);
    CircularBuffer cb_partial_obj(cb_partial);
    CircularBuffer cb_combined_obj(cb_combined);

    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;

    // Valid rows in the last H tile (for padding exclusion).
    constexpr uint32_t last_tile_rows = ((H % tile_height) == 0) ? tile_height : (H % tile_height);

    // Population variance: scale_idx = H-1 gives reciprocal 1/H.
    // Bessel's correction is applied later by the writer kernel.
    constexpr uint32_t scale_idx = H - 1;

    compute_kernel_hw_startup(cb_in, cb_partial);
    pack_reconfig_data_format(cb_partial);

    uint32_t num_outputs = NC_per_core / reduce_batch_size;

    for (uint32_t out = 0; out < num_outputs; ++out) {
        // --- Phase 1: H-reduce all columns for reduce_batch_size NC slices ---
        // Restore unpacker to cb_in's format after Phase 2 set it to
        // cb_combined (Float32).
        reconfig_data_format_srca(cb_in);
        for (uint32_t b = 0; b < reduce_batch_size; ++b) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                // H-reduce one column of Ht tiles.

                // start_N is the cumulative row count across tiles processed so far;
                // passed to the Welford LLK so it can compute the correct 1/(N+1) reciprocal
                // for each row's running-mean update.
                uint32_t start_N = 0;
                welford_init();

                // Process one tile-column along H while keeping a single running Welford state.
                // Welford's running accumulators (mean in LREG4, M2 in LREG5)
                // live in SFPU local registers (LREGs), which are separate
                // from the DST register file.  This means the Welford state survives
                // across tile_regs_release/acquire cycles -- only DST contents are
                // affected by the handshake, not the SFPU accumulators.
                //
                // Only SFPU-compatible operations are used (copy_tile + welford_update), so no
                // configuration conflict exists and the entire loop can run in a single DST
                // window (one acquire before the loop, one commit after the last tile). Only the
                // final iteration needs to expose result tiles to PACK.
                //
                // Per iteration:
                // - For all non-last H tiles, welford_update(input_dst, start_N, ...) consumes the full
                //   tile and updates the running mean/M2 using start_N as the global element offset.
                // - For the last H tile, welford_update_rows(..., last_tile_rows, ...) ignores padded
                //   rows so only valid elements participate in the statistics.
                // - welford_finalize_to_row(mean_dst, scale_idx, ...) converts M2 into variance and
                //   writes final mean/variance tiles into DST.
                // - start_N advances by one tile height each iteration so Welford sees the correct
                //   element count / divisor progression across the whole H reduction.
                copy_init(cb_in);
                tile_regs_acquire();

                // Welford SFPU state (running mean in LREG4, M2 in LREG5)
                // persists across DST cycles because LREGs are separate from
                // the DST register file managed by tile_regs_acquire/release.
                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    cb_in_obj.wait_front(onetile);
                    // copy_tile reads cb_in. For FP32 input, c_0 carries UnpackToDestFp32
                    // so the FP32 mantissa is preserved into DEST.
                    copy_tile(cb_in, 0, input_dst);
                    cb_in_obj.pop_front(onetile);

                    if (ht < (Ht - 1)) {
                        welford_update<0>(input_dst, start_N, {});
                    } else {
                        // Last tile: process only valid rows, then finalize.
                        welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                        // Finalize to row format: 32 per-column (mean, var) values
                        // stored in tile row 0 (across Face 0 and Face 1).
                        // welford_finalize_to_row applies SFPTRANSP to convert from
                        // SFPU lane order to tile column order; the "raw face" variant
                        // (welford_finalize_to_face) skips this and stores in lane
                        // order, which is NOT the same as tile column order.
                        // Population variance (scale_idx = H-1); Bessel's correction
                        // is applied by the writer kernel after W-combine.
                        welford_finalize_to_row<0>(mean_dst, scale_idx, {});
                        tile_regs_commit();
                    }
                    start_N += tile_height;
                }

                // Pack mean (DST[1]) and var (DST[2]) tiles to cb_partial.
                cb_partial_obj.reserve_back(2);
                tile_regs_wait();
                pack_reconfig_data_format(cb_partial);
                pack_tile_block(mean_dst, cb_partial, 2);
                tile_regs_release();
                cb_partial_obj.push_back(2);
            }
        }

        // --- Phase 2: Read combined scalar from writer, apply sqrt if std, post-mul, repack ---
        // The writer W-combines all per-column partials from Phase 1 into a
        // single Float32 scalar tile in cb_combined (with Bessel's correction
        // already applied).  We unpack it, apply sqrt_tile for std, apply the
        // user scalar, and re-pack into cb_out using the packer, which converts
        // to the output data format (handles BFLOAT8_B and all other formats).
        cb_combined_obj.wait_front(onetile);
        // Explicit srca reconfig is required because the unpacker was last
        // configured for cb_in's format (e.g. Float16_b) during Phase 1.
        // cb_combined uses Float32, so the unpacker must be reconfigured.
        reconfig_data_format_srca(cb_combined);
        tile_regs_acquire();
        copy_init(cb_combined);
        copy_tile(cb_combined, 0, input_dst);
        if constexpr (is_std) {
            sqrt_tile_init();
            sqrt_tile(input_dst);
        }
#ifdef WELFORD_POST_MUL
        // Apply the user scalar to the reduced output: var(s*x)=s^2 var(x), std(s*x)=|s| std(x).
        // mul_unary_tile is an SFPU op on DEST at full fp32 precision (issue #45222).
        binop_with_scalar_tile_init();
        mul_unary_tile(input_dst, post_mul_scaler_bits);
#endif
        tile_regs_commit();
        cb_combined_obj.pop_front(onetile);

        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(input_dst, cb_out);
        tile_regs_release();
        cb_out_obj.push_back(onetile);
    }
}
