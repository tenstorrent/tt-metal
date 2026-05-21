// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford HW-dimension reduction kernel (compute side), ported to Metal 2.0.
//
// Phase 1 (per output): For each of reduce_batch_size NC slices,
// H-reduces each of Wt columns using the Welford LLK, finalizes to
// row format (welford_finalize_to_row) and packs the mean+var tile
// pair to dfb::partial for the writer kernel to W-combine using the
// parallel Welford merge formula.
//
// Phase 2 (per output): Reads the combined Float32 scalar tile from
// dfb::combined (produced by the writer after W-combining all partials
// and applying Bessel's correction), applies sqrt_tile when computing
// std, and re-packs to dfb::output in the output data format.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Runtime arg: total number of NC slices this core must process.
    auto NC_per_core = get_arg(args::NC_per_core);

    // Compile-time args:
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr uint32_t Wt = get_arg(args::Wt);
    // do_scale is gated by the DO_SCALE define (host-side, in welford_reduce factory).
    constexpr uint32_t reduce_batch_size = get_arg(args::reduce_batch_size);
    constexpr bool is_std = get_arg(args::is_std) != 0;

    constexpr uint32_t onetile = 1;

    DataflowBuffer cb_in_obj(dfb::input);
    DataflowBuffer cb_out_obj(dfb::output);
    DataflowBuffer cb_partial_obj(dfb::partial);
    DataflowBuffer cb_combined_obj(dfb::combined);
    // Scaler DFB is bound on the compute kernel as CONSUMER unconditionally so the DFB
    // has a balanced producer/consumer pair; the value is only used when do_scale.
    DataflowBuffer cb_scalar_obj(dfb::scaler);

    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;

    // Valid rows in the last H tile (for padding exclusion).
    constexpr uint32_t last_tile_rows = ((H % tile_height) == 0) ? tile_height : (H % tile_height);

    // Population variance: scale_idx = H-1 gives reciprocal 1/H.
    // Bessel's correction is applied later by the writer kernel.
    constexpr uint32_t scale_idx = H - 1;

    compute_kernel_hw_startup(dfb::input, dfb::partial);
    pack_reconfig_data_format(dfb::partial);

    // Wait on scaler unconditionally to balance the producer (reader's prepare_reduce_scaler).
    // The value is only used when do_scale.
    cb_scalar_obj.wait_front(onetile);

    uint32_t num_outputs = NC_per_core / reduce_batch_size;

    for (uint32_t out = 0; out < num_outputs; ++out) {
        // --- Phase 1: H-reduce all columns for reduce_batch_size NC slices ---
        // Restore unpacker to dfb::input's format after Phase 2 set it to
        // dfb::combined (Float32).
        reconfig_data_format_srca(dfb::input);
        for (uint32_t b = 0; b < reduce_batch_size; ++b) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                // H-reduce one column of Ht tiles.
                uint32_t start_N = 0;
                welford_init();

#ifndef DO_SCALE
                copy_tile_to_dst_init_short(dfb::input);
                tile_regs_acquire();
#endif

                for (uint32_t ht = 0; ht < Ht; ++ht) {
#ifdef DO_SCALE
                    cb_in_obj.wait_front(onetile);
                    tile_regs_acquire();
                    mul_tiles_bcast_scalar_init_short(dfb::input, dfb::scaler);
                    mul_tiles_bcast_scalar(dfb::input, dfb::scaler, 0, 0, input_dst);
                    cb_in_obj.pop_front(1);

                    // Reconfigure the compute setup from scalar-multiply mode back to the
                    // SFPU state that Welford expects.
                    welford_reinit(dfb::input);
#else
                    cb_in_obj.wait_front(onetile);
                    copy_tile(dfb::input, 0, input_dst);
                    cb_in_obj.pop_front(onetile);
#endif

                    if (ht < (Ht - 1)) {
                        welford_update<0>(input_dst, start_N, {});
#ifdef DO_SCALE
                        tile_regs_commit();
                        tile_regs_wait();
                        tile_regs_release();
#endif
                    } else {
                        welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                        welford_finalize_to_row<0>(mean_dst, scale_idx, {});
                        tile_regs_commit();
                    }
                    start_N += tile_height;
                }

                // Pack mean (DST[1]) and var (DST[2]) tiles to dfb::partial.
                cb_partial_obj.reserve_back(2);
                tile_regs_wait();
                pack_reconfig_data_format(dfb::partial);
                pack_tile_block(mean_dst, dfb::partial, 2);
                tile_regs_release();
                cb_partial_obj.push_back(2);
            }
        }

        // --- Phase 2: Read combined scalar from writer, apply sqrt if std, repack ---
        cb_combined_obj.wait_front(onetile);
        reconfig_data_format_srca(dfb::combined);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(dfb::combined);
        copy_tile(dfb::combined, 0, input_dst);
        if constexpr (is_std) {
            sqrt_tile_init();
            sqrt_tile(input_dst);
        }
        tile_regs_commit();
        cb_combined_obj.pop_front(onetile);

        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::output);
        pack_tile(input_dst, dfb::output);
        tile_regs_release();
        cb_out_obj.push_back(onetile);
    }
}
