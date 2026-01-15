// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
void kernel_main() {
    constexpr int onetile = 1;
    ArgFetcher arg_fetcher;
    const uint32_t batch_num = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t Ht = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t Wt = arg_fetcher.get_next_arg_val<uint32_t>();
    const bool do_mask_h = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);
    const bool do_mask_w = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_1;
    constexpr auto cb_mask_h_w = tt::CBIndex::c_2;
    constexpr auto cb_intermed0 = tt::CBIndex::c_24;
    constexpr auto cb_intermed1 = tt::CBIndex::c_25;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(cb_in0, cb_in0, cb_out0);
    cb_wait_front(cb_scaler, onetile);

    if (do_mask_h || do_mask_w) {
        cb_wait_front(cb_mask_h_w, onetile * 2);
    }

    uint32_t num_tiles = batch_num * Ht * Wt;
    uint32_t num_tile_done = 0;
    for (uint32_t b = 0; b < batch_num; ++b) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                bool last_row = (ht == Ht - 1);
                bool last_col = (wt == Wt - 1);
                bool last_out = (num_tile_done == num_tiles - 1);
                bool do_mask = (do_mask_h && last_row) || (do_mask_w && last_col);

                auto cb_reduce = cb_in0;
                if (do_mask) {
                    // get tile from reader and apply mask
                    cb_wait_front(cb_in0, onetile);
                    tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
                    reconfig_data_format_srca(cb_in0);
#endif
                    copy_tile_to_dst_init_short(cb_in0);
                    copy_tile(cb_in0, 0, dst0);

                    if (do_mask_h && last_row) {
#if defined FP32_DEST_ACC_EN
                        reconfig_data_format_srca(cb_mask_h_w);
#endif
                        copy_tile_to_dst_init_short(cb_mask_h_w);
                        copy_tile(cb_mask_h_w, 0, dst1);
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }

                    if (do_mask_w && last_col) {
#if defined FP32_DEST_ACC_EN
                        reconfig_data_format_srca(cb_mask_h_w);
#endif
                        copy_tile_to_dst_init_short(cb_mask_h_w);
                        copy_tile(cb_mask_h_w, 1, dst1);
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    cb_reserve_back(cb_intermed0, onetile);
#if defined FP32_DEST_ACC_EN
                    pack_reconfig_data_format(cb_intermed0);
#endif
                    pack_tile(dst0, cb_intermed0);
                    cb_push_back(cb_intermed0, onetile);
                    tile_regs_release();

                    cb_pop_front(cb_in0, onetile);
                    cb_reduce = cb_intermed0;
                }

                auto output_cb = last_out ? cb_out0 : cb_intermed1;
                compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
                    cb_reduce,
                    cb_scaler,
                    output_cb,
                    compute_kernel_lib::TileShape::single(),
                    {},  // layout (use default)
                    compute_kernel_lib::Accumulate::at(cb_intermed1, num_tile_done));

                num_tile_done++;
            }
        }
    }
}
