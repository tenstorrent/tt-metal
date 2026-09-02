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
#include "experimental/kernel_args.h"

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
    std::uint32_t NC_per_core = get_arg(args::NC_per_core);

    // Compile-time args:
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto H = get_arg(args::H);
    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr auto Wt = get_arg(args::Wt);
#ifdef WELFORD_POST_MUL
    // Packed fp32 post-multiplier applied to the reduced output via mul_unary_tile (SFPU).
    // For var this is scalar^2, for std it is |scalar| (see welford_reduce_program_factory).
    constexpr auto post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif
    constexpr auto reduce_batch_size = get_arg(args::reduce_batch_size);
    constexpr bool is_std = get_arg(args::is_std) != 0;
    constexpr auto two_pass_mean_reciprocal = get_arg(args::two_pass_mean_reciprocal);
    constexpr auto two_pass_variance_reciprocal = get_arg(args::two_pass_variance_reciprocal);
#ifdef WELFORD_SFPU_LEAF_COMBINE
    constexpr auto welford_leaf_reciprocal = get_arg(args::welford_leaf_reciprocal);
#endif

    constexpr std::uint32_t onetile = 1;

    // For FP32 input dfb::in is flagged UnpackToDest (program factory), preserving FP32
    // mantissa for the copy_tile -> statistics SFPU consumer path. BF16 input: UnpackToSrc.
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_partial(dfb::partial);
    DataflowBuffer dfb_combined(dfb::combined);

    constexpr std::uint32_t input_dst = 0;
    constexpr std::uint32_t mean_dst = 1;
    constexpr std::uint32_t var_dst = 2;
    constexpr std::uint32_t retained_input_dst = 3;

    // Valid rows in the last H tile (for padding exclusion).
    constexpr std::uint32_t last_tile_rows = ((H % tile_height) == 0) ? tile_height : (H % tile_height);

    compute_kernel_hw_startup(dfb::in, dfb::partial);
    pack_reconfig_data_format(dfb::partial);

    std::uint32_t num_outputs = NC_per_core / reduce_batch_size;

    for (std::uint32_t out = 0; out < num_outputs; ++out) {
        // --- Phase 1: H-reduce all columns for reduce_batch_size NC slices ---
        // Restore unpacker to dfb::in's format after Phase 2 set it to
        // dfb::combined (Float32).
        reconfig_data_format_srca(dfb::in);
        for (std::uint32_t b = 0; b < reduce_batch_size; ++b) {
            for (std::uint32_t wt = 0; wt < Wt; ++wt) {
                copy_init(dfb::in);
                tile_regs_acquire();
                two_pass_stats_init_shifted();

                for (std::uint32_t ht = 0; ht < Ht; ++ht) {
#ifdef WELFORD_TWO_PASS_L1_REPLAY
                    dfb_in.wait_front(ht + 1);
                    copy_tile(dfb::in, ht, input_dst);
                    constexpr std::uint32_t stats_input_dst = input_dst;
#else
                    dfb_in.wait_front(onetile);
                    // Keep var_dst clean: finalization writes only the result rows, so
                    // parking pass-one input there would leak stale data into padding.
                    const std::uint32_t stats_input_dst =
                        ht < 2 ? (ht == 0 ? retained_input_dst : mean_dst) : input_dst;
                    copy_tile(dfb::in, 0, stats_input_dst);
                    dfb_in.pop_front(onetile);
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
                    copy_tile(dfb::in, ht, input_dst);
                    two_pass_stats_update_rows<true>(input_dst, 0, ht == Ht - 1 ? last_tile_rows : tile_height);
                }
                dfb_in.pop_front(Ht);
#else
                constexpr std::uint32_t num_front_retained = Ht < 2 ? Ht : 2;
                for (std::uint32_t ht = 0; ht < num_front_retained; ++ht) {
                    const std::uint32_t stats_input_dst = ht == 0 ? retained_input_dst : ht;
                    two_pass_stats_update_rows<true>(stats_input_dst, 0, ht == Ht - 1 ? last_tile_rows : tile_height);
                }
                if constexpr (Ht > num_front_retained) {
                    for (std::uint32_t ht = num_front_retained; ht < Ht - 1; ++ht) {
                        dfb_in.wait_front(onetile);
                        copy_tile(dfb::in, 0, retained_input_dst);
                        dfb_in.pop_front(onetile);
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
                dfb_partial.reserve_back(2);
                tile_regs_wait();
                pack_reconfig_data_format(dfb::partial);
#ifdef WELFORD_SFPU_LEAF_COMBINE
                pack_tile(retained_input_dst, dfb::partial);
                pack_tile(var_dst, dfb::partial);
#else
                pack_block(mean_dst, dfb::partial, 2);
#endif
                tile_regs_release();
                dfb_partial.push_back(2);
            }
        }

        // --- Phase 2: Read combined scalar from writer, apply sqrt if std, post-mul, repack ---
        // The writer W-combines all per-column partials from Phase 1 into a
        // single Float32 scalar tile in dfb::combined (with Bessel's correction
        // already applied).  We unpack it, apply sqrt_tile for std, apply the
        // user scalar, and re-pack into dfb::out using the packer, which converts
        // to the output data format (handles BFLOAT8_B and all other formats).
        dfb_combined.wait_front(onetile);
        // Explicit srca reconfig is required because the unpacker was last
        // configured for dfb::in's format (e.g. Float16_b) during Phase 1.
        // dfb::combined uses Float32, so the unpacker must be reconfigured.
        reconfig_data_format_srca(dfb::combined);
        tile_regs_acquire();
        copy_init(dfb::combined);
        copy_tile(dfb::combined, 0, input_dst);
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
        dfb_combined.pop_front(onetile);

        dfb_out.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::out);
        pack_tile(input_dst, dfb::out);
        tile_regs_release();
        dfb_out.push_back(onetile);
    }
}
