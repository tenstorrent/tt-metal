// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    // Carries the per-core work-split count (the host's num_cols_per_core_group_N), not a tile width.
    uint32_t Wt = get_arg(args::units_per_core);
    uint32_t NC = get_arg(args::NC);
    constexpr uint32_t origin_H = get_arg(args::origin_H);

    DataflowBuffer dfb_scaler_obj(dfb::scaler);
    DataflowBuffer dfb_mask_h_obj(dfb::mask_h);
    constexpr uint32_t TILE_H = 32;
    constexpr bool do_mask_h = (origin_H % TILE_H) != 0;

    compute_kernel_hw_startup(dfb::input, dfb::input, dfb::out);

    dfb_scaler_obj.wait_front(1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    const uint32_t mask_dst_idx = reduce_dst_idx + 1;

    if constexpr (do_mask_h) {
        dfb_mask_h_obj.wait_front(onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // tiles are expected to be coming in in NCWH order (H-contiguous)
            // reducing in W means out[0][w] = sum(h=0..H-1, in[h][w])
            // in this case we just sequentially add to accumulator all the H-tiles in a column
            constexpr bool is_h_single_tile = Ht == 1;

            // Phase 1: Reduce Ht-1 tiles into accumulator (if Ht > 1)
            if constexpr (!is_h_single_tile) {
                ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb::input, dfb::scaler, dfb::accum_dst>(
                    ckl::ReduceInputBlockShape::col(Ht - 1));
            }

            // Optional masking of last H tile.
            if constexpr (do_mask_h) {
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(onetile),
                    ckl::CopyTile<ckl::input(
                        dfb::input, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
                    ckl::CopyTile<
                        ckl::input(dfb::mask_h, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                        ckl::Dst::D1>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                    ckl::PackTile<ckl::output(
                        dfb::masked_input,
                        ckl::ReservePolicy::PerTile,
                        ckl::PushPolicy::PerTile,
                        kDataFormatReconfig)>{});

                // Phase 2 with masked input: Reduce final masked tile with accumulation
                ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb::masked_input, dfb::scaler, dfb::out>(
                    ckl::ReduceInputBlockShape::single(),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(dfb::accum_dst, is_h_single_tile ? 0 : 1));
            } else {
                // Phase 2 without masking: Reduce final tile with accumulation
                // - If Ht == 1 (single tile): iteration=0, no accumulator reload
                // - If Ht > 1 (multi-tile): iteration=1, reload accumulator from dfb::accum_dst
                ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb::input, dfb::scaler, dfb::out>(
                    ckl::ReduceInputBlockShape::single(),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(dfb::accum_dst, is_h_single_tile ? 0 : 1));
            }
        }
    }

    if constexpr (do_mask_h) {
        dfb_mask_h_obj.pop_front(onetile);
    }
    dfb_scaler_obj.pop_front(onetile);
}
