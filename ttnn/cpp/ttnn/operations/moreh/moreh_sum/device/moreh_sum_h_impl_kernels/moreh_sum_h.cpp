// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t Ht = get_arg(args::Ht);
    // Carries the per-core work-split count (the host's num_cols_per_core_group_N), not a tile width.
    uint32_t Wt = get_arg(args::units_per_core);
    uint32_t NC = get_arg(args::NC);
    constexpr uint32_t origin_H = get_arg(args::origin_H);

    DataflowBuffer dfb_input_obj(dfb::input);
    DataflowBuffer dfb_scaler_obj(dfb::scaler);
    DataflowBuffer dfb_mask_h_obj(dfb::mask_h);
    DataflowBuffer dfb_masked_input_obj(dfb::masked_input);
    constexpr uint32_t TILE_H = 32;
    constexpr bool do_mask_h = (origin_H % TILE_H) != 0;

    compute_kernel_hw_startup(dfb::input, dfb::input, dfb::out);

    dfb_scaler_obj.wait_front(1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    const uint32_t mask_dst_idx = reduce_dst_idx + 1;

    if (do_mask_h) {
        dfb_mask_h_obj.wait_front(onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // tiles are expected to be coming in in NCWH order (H-contiguous)
            // reducing in W means out[0][w] = sum(h=0..H-1, in[h][w])
            // in this case we just sequentially add to accumulator all the H-tiles in a column
            bool is_h_single_tile = (Ht == 1);

            // Phase 1: Reduce Ht-1 tiles into accumulator (if Ht > 1)
            if (!is_h_single_tile) {
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::input, dfb::scaler, dfb::accum_dst>(
                    compute_kernel_lib::ReduceInputBlockShape::col(Ht - 1));
            }

            // Optional masking of last H tile
            if constexpr (do_mask_h) {
                tile_regs_acquire();
                dfb_input_obj.wait_front(onetile);
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(dfb::input);
#endif
                copy_tile_to_dst_init_short(dfb::input);
                copy_tile(dfb::input, 0, reduce_dst_idx);
                copy_tile(dfb::mask_h, 0, mask_dst_idx);
                mask_tile_init();
                mask_tile(reduce_dst_idx, mask_dst_idx);
                tile_regs_commit();

                dfb_masked_input_obj.reserve_back(onetile);
                tile_regs_wait();
#if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(dfb::masked_input);
#endif
                pack_tile(reduce_dst_idx, dfb::masked_input);
                tile_regs_release();
                dfb_masked_input_obj.push_back(onetile);

                dfb_input_obj.pop_front(onetile);

                // Phase 2 with masked input: Reduce final masked tile with accumulation
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::masked_input, dfb::scaler, dfb::out>(
                    compute_kernel_lib::ReduceInputBlockShape::single(),
                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                    compute_kernel_lib::Accumulate::at(dfb::accum_dst, is_h_single_tile ? 0 : 1));
            } else {
                // Phase 2 without masking: Reduce final tile with accumulation
                // - If Ht == 1 (single tile): iteration=0, no accumulator reload
                // - If Ht > 1 (multi-tile): iteration=1, reload accumulator from the accum_dst DFB
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::input, dfb::scaler, dfb::out>(
                    compute_kernel_lib::ReduceInputBlockShape::single(),
                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                    compute_kernel_lib::Accumulate::at(dfb::accum_dst, is_h_single_tile ? 0 : 1));
            }
        }
    }

    if (do_mask_h) {
        dfb_mask_h_obj.pop_front(onetile);
    }
    dfb_scaler_obj.pop_front(onetile);
}
