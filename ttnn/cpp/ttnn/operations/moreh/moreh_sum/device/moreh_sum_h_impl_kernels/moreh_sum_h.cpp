// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_H = get_compile_time_arg_val(3);

    auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    constexpr auto cb_mask_h = tt::CBIndex::c_3;
    constexpr auto cb_accum_dst = tt::CBIndex::c_24;
    constexpr auto cb_masked_input = tt::CBIndex::c_25;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t TILE_H = 32;
    constexpr bool do_mask_h = (origin_H % TILE_H) != 0;

    binary_op_init_common(cb_input, cb_input, cb_out);

    cb_wait_front(cb_scaler, 1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    const uint32_t mask_dst_idx = reduce_dst_idx + 1;

    if (do_mask_h) {
        cb_wait_front(cb_mask_h, onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // tiles are expected to be coming in in NCWH order (H-contiguous)
            // reducing in W means out[0][w] = sum(h=0..H-1, in[h][w])
            // in this case we just sequentially add to accumulator all the H-tiles in a column
            cb_input = tt::CBIndex::c_0;
            bool is_h_single_tile = (Ht == 1);

            // Phase 1: Reduce Ht-1 tiles into accumulator (if Ht > 1)
            if (!is_h_single_tile) {
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
                    cb_input, cb_scaler, cb_accum_dst, compute_kernel_lib::TileShape::col(Ht - 1));
            }

            // Optional masking of last H tile
            if (do_mask_h) {
                tile_regs_acquire();
                cb_wait_front(cb_input, onetile);
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_input);
#endif
                copy_tile_to_dst_init_short(cb_input);
                copy_tile(cb_input, 0, reduce_dst_idx);
                copy_tile(cb_mask_h, 0, mask_dst_idx);
                mask_tile_init();
                mask_tile(reduce_dst_idx, mask_dst_idx);
                tile_regs_commit();

                cb_reserve_back(cb_masked_input, onetile);
                tile_regs_wait();
#if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(cb_masked_input);
#endif
                pack_tile(reduce_dst_idx, cb_masked_input);
                tile_regs_release();
                cb_push_back(cb_masked_input, onetile);

                cb_pop_front(cb_input, onetile);
                cb_input = cb_masked_input;
            }

            // Phase 2: Reduce final tile with accumulation
            // - If Ht == 1 (single tile): iteration=0, no accumulator reload
            // - If Ht > 1 (multi-tile): iteration=1, reload accumulator from cb_accum_dst
            compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
                cb_input,
                cb_scaler,
                cb_out,
                compute_kernel_lib::TileShape::single(),
                {},  // layout (use default)
                compute_kernel_lib::Accumulate::at(cb_accum_dst, is_h_single_tile ? 0 : 1));
        }
    }

    if (do_mask_h) {
        cb_pop_front(cb_mask_h, onetile);
    }
    cb_pop_front(cb_scaler, onetile);
}
