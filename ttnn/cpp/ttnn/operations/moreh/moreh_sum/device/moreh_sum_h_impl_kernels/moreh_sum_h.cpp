// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_H = get_compile_time_arg_val(3);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    DataflowBuffer cb_scaler_obj(cb_scaler);
    constexpr auto cb_mask_h = tt::CBIndex::c_3;
    DataflowBuffer cb_mask_h_obj(cb_mask_h);
    constexpr auto cb_accum_dst = tt::CBIndex::c_24;
    constexpr auto cb_masked_input = tt::CBIndex::c_25;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t TILE_H = 32;
    constexpr bool do_mask_h = (origin_H % TILE_H) != 0;

    binary_op_init_common(cb_input, cb_input, cb_out);

    cb_scaler_obj.wait_front(1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    const uint32_t mask_dst_idx = reduce_dst_idx + 1;

    if constexpr (do_mask_h) {
        cb_mask_h_obj.wait_front(onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // tiles are expected to be coming in in NCWH order (H-contiguous)
            // reducing in W means out[0][w] = sum(h=0..H-1, in[h][w])
            // in this case we just sequentially add to accumulator all the H-tiles in a column
            constexpr bool is_h_single_tile = Ht == 1;

            // Phase 1: Reduce Ht-1 tiles into accumulator (if Ht > 1)
            if constexpr (!is_h_single_tile) {
                ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_input, cb_scaler, cb_accum_dst>(
                    ckl::ReduceInputBlockShape::col(Ht - 1));
            }

            // Optional masking of last H tile.
            if constexpr (do_mask_h) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::CopyTile<ckl::input(cb_input, ckl::InputLifecycle::Streaming, kDataFormatReconfig)>{},
                    ckl::CopyTile<
                        ckl::input(cb_mask_h, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                        ckl::Dst::D1>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                    ckl::PackTile<ckl::output(
                        cb_masked_input, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

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
        cb_mask_h_obj.pop_front(onetile);
    }
    cb_scaler_obj.pop_front(onetile);
}
