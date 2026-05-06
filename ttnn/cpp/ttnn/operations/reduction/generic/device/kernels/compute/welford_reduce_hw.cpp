// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 Welford HW-dimension reduction compute kernel (compute side).
//
// Phase 1 (per output): For each of reduce_batch_size NC slices, H-reduces each
// of Wt columns using the Welford LLK, finalizes to row format and packs the
// mean+var tile pair to cb_partial for the writer kernel to W-combine.
// Phase 2 (per output): Reads the combined Float32 scalar tile from cb_combined
// (produced by the writer after W-combining all partials and applying Bessel's
// correction), applies sqrt_tile when computing std, and re-packs to cb_out in
// the output data format.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/dataflow_buffer.h"

void kernel_main() {
    // Runtime arg: total NC slices this core must process.
    const uint32_t NC_per_core = get_arg(args::NC_per_core);

    // Compile-time args.
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr bool do_scale = get_arg(args::do_scale) != 0;
    constexpr uint32_t reduce_batch_size = get_arg(args::reduce_batch_size);
    constexpr bool is_std = get_arg(args::is_std) != 0;

    constexpr uint32_t onetile = 1;

    experimental::DataflowBuffer cb_in_obj(dfb::input);
    experimental::DataflowBuffer cb_scalar_obj(dfb::scaler);
    experimental::DataflowBuffer cb_out_obj(dfb::output);
    experimental::DataflowBuffer cb_partial_obj(dfb::partial);
    experimental::DataflowBuffer cb_combined_obj(dfb::combined);

    const uint32_t cb_in = cb_in_obj.get_id();
    const uint32_t cb_scalar = cb_scalar_obj.get_id();
    const uint32_t cb_out = cb_out_obj.get_id();
    const uint32_t cb_partial = cb_partial_obj.get_id();
    const uint32_t cb_combined = cb_combined_obj.get_id();

    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;

    // Valid rows in the last H tile (for padding exclusion).
    constexpr uint32_t last_tile_rows = ((H % tile_height) == 0) ? tile_height : (H % tile_height);

    // Population variance: scale_idx = H-1 gives reciprocal 1/H.
    // Bessel's correction is applied later by the writer kernel.
    constexpr uint32_t scale_idx = H - 1;

    compute_kernel_hw_startup(cb_in, cb_partial);
    pack_reconfig_data_format(cb_partial);

    if constexpr (do_scale) {
        cb_scalar_obj.wait_front(onetile);
    }

    uint32_t num_outputs = NC_per_core / reduce_batch_size;

    for (uint32_t out = 0; out < num_outputs; ++out) {
        // Phase 1: H-reduce all columns for reduce_batch_size NC slices.
        // Restore unpacker to cb_in's format after Phase 2 set it to cb_combined (Float32).
        reconfig_data_format_srca(cb_in);
        for (uint32_t b = 0; b < reduce_batch_size; ++b) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                // H-reduce one column of Ht tiles.
                uint32_t start_N = 0;
                welford_init();

                if constexpr (!do_scale) {
                    copy_tile_to_dst_init_short(cb_in);
                    tile_regs_acquire();
                }

                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    if constexpr (do_scale) {
                        cb_in_obj.wait_front(onetile);
                        tile_regs_acquire();
                        mul_tiles_bcast_scalar_init_short(cb_in, cb_scalar);
                        mul_tiles_bcast_scalar(cb_in, cb_scalar, 0, 0, input_dst);
                        cb_in_obj.pop_front(1);
                        welford_reinit(cb_in);
                    } else {
                        cb_in_obj.wait_front(onetile);
                        copy_tile(cb_in, 0, input_dst);
                        cb_in_obj.pop_front(onetile);
                    }

                    if (ht < (Ht - 1)) {
                        welford_update<0>(input_dst, start_N, {});
                        if constexpr (do_scale) {
                            tile_regs_commit();
                            tile_regs_wait();
                            tile_regs_release();
                        }
                    } else {
                        welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                        // welford_finalize_to_row applies SFPTRANSP to convert from SFPU
                        // lane order to tile column order. Population variance (scale_idx
                        // = H-1); Bessel's correction is applied by the writer after W-combine.
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

        // Phase 2: Read combined scalar from writer, apply sqrt if std, repack.
        cb_combined_obj.wait_front(onetile);
        // Explicit srca reconfig: unpacker was last configured for cb_in's format
        // (Float16_b) during Phase 1; cb_combined uses Float32, so reconfigure.
        reconfig_data_format_srca(cb_combined);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_combined);
        copy_tile(cb_combined, 0, input_dst);
        if constexpr (is_std) {
            sqrt_tile_init();
            sqrt_tile(input_dst);
        }
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
