// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford HW-dimension reduction kernel (compute side).
//
// Phase 1 (per output): For each of reduce_batch_size NC slices,
// H-reduces each of Wt columns using two-pass statistics and packs the
// mean+variance tile pair to dfb_partial for the writer kernel to W-combine using the
// parallel Welford merge formula.
//
// Phase 2 (per output): Reads the combined Float32 scalar tile from
// dfb_combined (produced by the writer after W-combining all partials
// and applying Bessel's correction), applies sqrt_tile when computing
// std, applies the user scalar via SFPU post-multiplication, and
// re-packs to dfb_out in the output data format.  This ensures
// the packer hardware handles format conversion (required for
// BFLOAT8_B and for matching the output dtype to the input dtype).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"

#ifdef WELFORD_SFPU_LEAF_COMBINE
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#endif

#if defined(WELFORD_POST_MUL) || defined(WELFORD_SFPU_LEAF_COMBINE)
// SFPU multiply-by-scalar (mul_unary_tile) applied to the reduced output. See issue #45222.
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    // Runtime arg: total number of NC slices this core must process.
    std::uint32_t NC_per_core = get_arg_val<std::uint32_t>(0);

    // Compile-time args:
    constexpr std::uint32_t Ht = get_compile_time_arg_val(0);
    constexpr std::uint32_t H = get_compile_time_arg_val(1);
    constexpr std::uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr std::uint32_t Wt = get_compile_time_arg_val(3);
#ifdef WELFORD_POST_MUL
    // Packed fp32 post-multiplier applied to the reduced output via mul_unary_tile (SFPU).
    // For var this is scalar^2, for std it is |scalar| (see welford_reduce_program_factory).
    constexpr std::uint32_t post_mul_scaler_bits = get_compile_time_arg_val(4);
#endif
    constexpr std::uint32_t reduce_batch_size = get_compile_time_arg_val(5);
    constexpr bool is_std = get_compile_time_arg_val(6) != 0;
    constexpr std::uint32_t two_pass_mean_reciprocal = get_named_compile_time_arg_val("two_pass_mean_reciprocal");
    constexpr std::uint32_t two_pass_variance_reciprocal =
        get_named_compile_time_arg_val("two_pass_variance_reciprocal");
#ifdef WELFORD_SFPU_LEAF_COMBINE
    constexpr std::uint32_t welford_leaf_reciprocal = get_named_compile_time_arg_val("welford_leaf_reciprocal");
#endif

    constexpr std::uint32_t onetile = 1;

    // dfb_in: For FP32 input it is flagged UnpackToDestFp32 (program factory), preserving FP32
    // mantissa for the copy_tile -> welford SFPU consumer path. BF16 input: Default.
    constexpr auto dfb_in = tt::CBIndex::c_0;
    // Final output CB (output data format), consumed by the writer for NOC write.
    constexpr auto dfb_out = tt::CBIndex::c_16;
    // Intermediate CB for mean+var tile pairs, consumed by writer kernel.
    constexpr auto dfb_partial = tt::CBIndex::c_21;
    // Combined scalar result from the writer kernel (Float32).
    constexpr auto dfb_combined = tt::CBIndex::c_22;

    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_out_obj(dfb_out);
    DataflowBuffer dfb_partial_obj(dfb_partial);
    DataflowBuffer dfb_combined_obj(dfb_combined);

    constexpr std::uint32_t input_dst = 0;
    constexpr std::uint32_t mean_dst = 1;
    constexpr std::uint32_t var_dst = 2;
    constexpr std::uint32_t retained_input_dst = 3;

    // Valid rows in the last H tile (for padding exclusion).
    constexpr std::uint32_t last_tile_rows = ((H % tile_height) == 0) ? tile_height : (H % tile_height);

    compute_kernel_hw_startup(dfb_in, dfb_partial);
    pack_reconfig_data_format(dfb_partial);

    std::uint32_t num_outputs = NC_per_core / reduce_batch_size;

    for (std::uint32_t out = 0; out < num_outputs; ++out) {
        // --- Phase 1: H-reduce all columns for reduce_batch_size NC slices ---
        // Restore unpacker to dfb_in's format after Phase 2 set it to
        // dfb_combined (Float32).
        reconfig_data_format_srca(dfb_in);
        for (std::uint32_t b = 0; b < reduce_batch_size; ++b) {
            for (std::uint32_t wt = 0; wt < Wt; ++wt) {
                copy_init(dfb_in);
                tile_regs_acquire();
                two_pass_stats_init_shifted();

                for (std::uint32_t ht = 0; ht < Ht; ++ht) {
#ifdef WELFORD_TWO_PASS_L1_REPLAY
                    dfb_in_obj.wait_front(ht + 1);
                    copy_tile(dfb_in, ht, input_dst);
                    constexpr std::uint32_t stats_input_dst = input_dst;
#else
                    dfb_in_obj.wait_front(onetile);
                    const std::uint32_t stats_input_dst = ht < 3 ? (ht == 0 ? retained_input_dst : ht) : input_dst;
                    copy_tile(dfb_in, 0, stats_input_dst);
                    dfb_in_obj.pop_front(onetile);
#endif
                    if (ht == 0) {
                        two_pass_stats_update_shifted_rows<false, true>(
                            stats_input_dst, 0, ht == Ht - 1 ? last_tile_rows : tile_height);
                    } else {
                        two_pass_stats_update_shifted_rows<false>(
                            stats_input_dst, 0, ht == Ht - 1 ? last_tile_rows : tile_height);
                    }
                }
                two_pass_stats_finish_shifted_mean(two_pass_mean_reciprocal);

#ifdef WELFORD_TWO_PASS_L1_REPLAY
                for (std::uint32_t ht = 0; ht < Ht; ++ht) {
                    copy_tile(dfb_in, ht, input_dst);
                    two_pass_stats_update_rows<true>(input_dst, 0, ht == Ht - 1 ? last_tile_rows : tile_height);
                }
                dfb_in_obj.pop_front(Ht);
#else
                constexpr std::uint32_t num_front_retained = Ht < 3 ? Ht : 3;
                for (std::uint32_t ht = 0; ht < num_front_retained; ++ht) {
                    const std::uint32_t stats_input_dst = ht == 0 ? retained_input_dst : ht;
                    two_pass_stats_update_rows<true>(stats_input_dst, 0, ht == Ht - 1 ? last_tile_rows : tile_height);
                }
                if constexpr (Ht > num_front_retained) {
                    for (std::uint32_t ht = num_front_retained; ht < Ht - 1; ++ht) {
                        dfb_in_obj.wait_front(onetile);
                        copy_tile(dfb_in, 0, retained_input_dst);
                        dfb_in_obj.pop_front(onetile);
                        two_pass_stats_update_rows<true>(retained_input_dst, 0, tile_height);
                    }
                    two_pass_stats_update_rows<true>(input_dst, 0, last_tile_rows);
                }
#endif
                two_pass_stats_finalize_to_row(mean_dst, two_pass_variance_reciprocal);

#ifdef WELFORD_SFPU_LEAF_COMBINE
                // Collapse the 32 equal-count column statistics into one stable
                // leaf on SFPU. The writer then merges one record per input tile
                // instead of executing soft-float arithmetic for every column.
                copy_dest_values_init();
                copy_dest_values<DataFormat::Float32>(mean_dst, retained_input_dst);

                sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
                sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(
                    retained_input_dst, /*ct_dim=*/1, /*rt_dim=*/1);
                binop_with_scalar_tile_init();
                mul_unary_tile(retained_input_dst, welford_leaf_reciprocal);

                sfpu_bcast_col_init();
                sfpu_sub_bcast_col(mean_dst, retained_input_dst);
                square_tile_init();
                square_tile(mean_dst);
                add_binary_tile_init();
                add_binary_tile(var_dst, mean_dst, var_dst);

                sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
                sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(
                    var_dst, /*ct_dim=*/1, /*rt_dim=*/1);
#endif
                tile_regs_commit();

                // Pack mean (DST[1]) and var (DST[2]) tiles to dfb_partial.
                dfb_partial_obj.reserve_back(2);
                tile_regs_wait();
                pack_reconfig_data_format(dfb_partial);
#ifdef WELFORD_SFPU_LEAF_COMBINE
                pack_tile(retained_input_dst, dfb_partial);
                pack_tile(var_dst, dfb_partial);
#else
                pack_block(mean_dst, dfb_partial, 2);
#endif
                tile_regs_release();
                dfb_partial_obj.push_back(2);
            }
        }

        // --- Phase 2: Read combined scalar from writer, apply sqrt if std, post-mul, repack ---
        // The writer W-combines all per-column partials from Phase 1 into a
        // single Float32 scalar tile in dfb_combined (with Bessel's correction
        // already applied).  We unpack it, apply sqrt_tile for std, apply the
        // user scalar, and re-pack into dfb_out using the packer, which converts
        // to the output data format (handles BFLOAT8_B and all other formats).
        dfb_combined_obj.wait_front(onetile);
        // Explicit srca reconfig is required because the unpacker was last
        // configured for dfb_in's format (e.g. Float16_b) during Phase 1.
        // dfb_combined uses Float32, so the unpacker must be reconfigured.
        reconfig_data_format_srca(dfb_combined);
        tile_regs_acquire();
        copy_init(dfb_combined);
        copy_tile(dfb_combined, 0, input_dst);
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
        dfb_combined_obj.pop_front(onetile);

        dfb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb_out);
        pack_tile(input_dst, dfb_out);
        tile_regs_release();
        dfb_out_obj.push_back(onetile);
    }
}
