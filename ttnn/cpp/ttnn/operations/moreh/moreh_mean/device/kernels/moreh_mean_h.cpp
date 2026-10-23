// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/mask.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const auto Ht = get_arg(args::Ht);
    const auto Wt = get_arg(args::units_per_core);  // Per-core output-column count, not tile width.
    const auto NC = get_arg(args::NC);
    constexpr uint32_t origin_H = get_arg(args::origin_H);

    DataflowBuffer dfb_scaler_obj(dfb::scaler);
    DataflowBuffer dfb_mask_h_obj(dfb::mask_h);
    constexpr bool do_mask_h = (origin_H % TILE_HEIGHT) != 0;

    compute_kernel_hw_startup(dfb::input, dfb::input, dfb::out);

    dfb_scaler_obj.wait_front(1);  // scaler tile from the reader

    constexpr int onetile = 1;

    if constexpr (do_mask_h) {
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
                ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb::input, dfb::scaler, dfb::accum_dst>(
                    ckl::ReduceInputBlockShape::col(Ht - 1));
            }

            // Optional masking of last H tile
            if constexpr (do_mask_h) {
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(onetile),
                    ckl::CopyTile<ckl::input(dfb::input)>{},
                    ckl::CopyTile<ckl::input(dfb::mask_h, ckl::WaitPolicy::None, ckl::PopPolicy::None), ckl::Dst::D1>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                    ckl::PackTile<ckl::output(dfb::masked_input)>{});

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
