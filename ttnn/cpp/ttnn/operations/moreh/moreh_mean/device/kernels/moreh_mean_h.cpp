// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/mask.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const auto Ht = get_arg(args::Ht);
    const auto Wt = get_arg(args::units_per_core);
    const auto NC = get_arg(args::NC);
    constexpr uint32_t origin_H = get_arg(args::origin_H);

    constexpr auto cb_input = dfb::input;
    constexpr auto cb_scaler = dfb::scaler;
    DataflowBuffer dfb_scaler_obj(cb_scaler);
    constexpr auto cb_mask_h = dfb::mask_h;
    DataflowBuffer dfb_mask_h_obj(cb_mask_h);
    constexpr auto cb_accum_dst = dfb::accum_dst;
    constexpr auto cb_masked_input = dfb::masked_input;
    constexpr auto cb_out = dfb::out;
    constexpr bool do_mask_h = (origin_H % TILE_HEIGHT) != 0;

    compute_kernel_hw_startup(cb_input, cb_input, cb_out);

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
                ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_input, cb_scaler, cb_accum_dst>(
                    ckl::ReduceInputBlockShape::col(Ht - 1));
            }

            // Optional masking of last H tile
            if constexpr (do_mask_h) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::CopyTile<ckl::input(cb_input)>{},
                    ckl::CopyTile<ckl::input(cb_mask_h, ckl::WaitPolicy::None, ckl::PopPolicy::None), ckl::Dst::D1>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                    ckl::PackTile<ckl::output(cb_masked_input)>{});

                // Phase 2 with masked input: Reduce final masked tile with accumulation
                ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_masked_input, cb_scaler, cb_out>(
                    ckl::ReduceInputBlockShape::single(),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_accum_dst, is_h_single_tile ? 0 : 1));
            } else {
                // Phase 2 without masking: Reduce final tile with accumulation
                // - If Ht == 1 (single tile): iteration=0, no accumulator reload
                // - If Ht > 1 (multi-tile): iteration=1, reload accumulator from cb_accum_dst
                ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_input, cb_scaler, cb_out>(
                    ckl::ReduceInputBlockShape::single(),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_accum_dst, is_h_single_tile ? 0 : 1));
            }
        }
    }

    if constexpr (do_mask_h) {
        dfb_mask_h_obj.pop_front(onetile);
    }
    dfb_scaler_obj.pop_front(onetile);
}
