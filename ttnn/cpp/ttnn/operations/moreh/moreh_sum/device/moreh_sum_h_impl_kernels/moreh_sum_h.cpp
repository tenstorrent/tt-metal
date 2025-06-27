// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
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
            if (!is_h_single_tile) {
                tile_regs_acquire();
                for (uint32_t ht = 0; ht < Ht - 1; ++ht) {
                    cb_wait_front(cb_input, onetile);

#if defined FP32_DEST_ACC_EN
                    reconfig_data_format(cb_input, cb_scaler);
#endif
                    reduce_init(cb_input, cb_scaler, cb_accum_dst);
                    reduce_tile(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
                    reduce_uninit();

                    cb_pop_front(cb_input, onetile);
                }
                tile_regs_commit();
                cb_reserve_back(cb_accum_dst, onetile);
                tile_regs_wait();
#if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(cb_accum_dst);
#endif
                pack_tile(reduce_dst_idx, cb_accum_dst);
                tile_regs_release();
                cb_push_back(cb_accum_dst, onetile);
            }

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

            tile_regs_acquire();
            cb_wait_front(cb_input, onetile);
            if (!is_h_single_tile) {
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_accum_dst);
#endif
                cb_wait_front(cb_accum_dst, onetile);
                copy_tile_to_dst_init_short(cb_accum_dst);
                copy_tile(cb_accum_dst, 0, reduce_dst_idx);
            }

#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_input, cb_scaler);
#endif
            reduce_init(cb_input, cb_scaler, cb_out);
            reduce_tile(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
            reduce_uninit();
            tile_regs_commit();

            cb_reserve_back(cb_out, onetile);
            tile_regs_wait();
#if defined FP32_DEST_ACC_EN
            pack_reconfig_data_format(cb_out);
#endif
            pack_tile(reduce_dst_idx, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, onetile);

            cb_pop_front(cb_input, onetile);
            if (!is_h_single_tile) {
                cb_pop_front(cb_accum_dst, onetile);
            }
        }
    }

    if (do_mask_h) {
        cb_pop_front(cb_mask_h, onetile);
    }
    cb_pop_front(cb_scaler, onetile);
}
}  // namespace NAMESPACE
