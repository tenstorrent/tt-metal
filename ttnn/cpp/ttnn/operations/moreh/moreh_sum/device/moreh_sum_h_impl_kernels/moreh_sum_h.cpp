// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_H = get_compile_time_arg_val(3);

    constexpr auto cb_input = tt::CBIndex::c_0;
    CircularBuffer cb_input_obj(cb_input);
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    CircularBuffer cb_scaler_obj(cb_scaler);
    constexpr auto cb_mask_h = tt::CBIndex::c_3;
    CircularBuffer cb_mask_h_obj(cb_mask_h);
    constexpr auto cb_accum_dst = tt::CBIndex::c_24;
    constexpr auto cb_masked_input = tt::CBIndex::c_25;
    CircularBuffer cb_masked_input_obj(cb_masked_input);
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t TILE_H = 32;
    constexpr bool do_mask_h = (origin_H % TILE_H) != 0;

    binary_op_init_common(cb_input, cb_input, cb_out);

    cb_scaler_obj.wait_front(1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    const uint32_t mask_dst_idx = reduce_dst_idx + 1;

    if (do_mask_h) {
        cb_mask_h_obj.wait_front(onetile);
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

            // Optional masking of last H tile.
            if constexpr (do_mask_h) {
                // CopyTile<cb_input> + CopyTile<cb_mask_h, D1> + Mask + PackTile.
                // Reconfig: original uses `copy_tile_to_dst_init_short` (no _with_dt)
                // and plain pack_tile, with a single FP32_DEST_ACC_EN-guarded
                // reconfig_data_format_srca(cb_input) on entry. The chain's
                // per-element fold emits reconfig per CopyTile (Input mode) which
                // matches the FP32 path and is a no-op transition for bf16.
                // pack_reconfig is also FP32-only in original; enabling output reconfig
                // emits unconditional pack reconfig — same effective behavior since
                // chain's prev-CB elision handles the no-op case at compile time.
                // cb_input InputLifecycle::Streaming; cb_mask_h InputLifecycle::CallerManaged + Scalar (held outside);
                // cb_masked_input OutputLifecycle::Streaming.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::CopyTile<cb_input>{},
                    ckl::CopyTile<cb_mask_h, ckl::Dst::D1, ckl::input(ckl::InputLifecycle::CallerManaged)>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                    ckl::PackTile<cb_masked_input>{});

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

    if (do_mask_h) {
        cb_mask_h_obj.pop_front(onetile);
    }
    cb_scaler_obj.pop_front(onetile);
}
