// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr int onetile = 1;
    ArgFetcher arg_fetcher;
    const uint32_t batch_num = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t Ht = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t Wt_per_core = arg_fetcher.get_next_arg_val<uint32_t>();
    const bool do_mask_h = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);
    const bool do_mask_w = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    DataflowBuffer dfb_in0_obj(cb_in0);
    constexpr auto cb_scaler = tt::CBIndex::c_1;
    DataflowBuffer dfb_scaler_obj(cb_scaler);
    constexpr auto cb_mask_h_w = tt::CBIndex::c_2;
    DataflowBuffer dfb_mask_h_w_obj(cb_mask_h_w);
    constexpr auto cb_intermed0 = tt::CBIndex::c_24;
    DataflowBuffer dfb_intermed0_obj(cb_intermed0);
    constexpr auto cb_intermed1 = tt::CBIndex::c_25;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(cb_in0, cb_in0, cb_out0);
    dfb_scaler_obj.wait_front(onetile);

    if (do_mask_h || do_mask_w) {
        dfb_mask_h_w_obj.wait_front(onetile * 2);
    }

    uint32_t num_tiles = batch_num * Ht;
    for (uint32_t wt = 0; wt < Wt_per_core; ++wt) {
        uint32_t num_tile_done = 0;
        for (uint32_t b = 0; b < batch_num; ++b) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                bool last_row = (ht == Ht - 1);
                bool last_col = (wt == Wt_per_core - 1);
                bool last_out = (num_tile_done == num_tiles - 1);
                bool do_mask = (do_mask_h && last_row) || (do_mask_w && last_col);

                if (do_mask) {
                    // get tile from reader and apply mask
                    dfb_in0_obj.wait_front(onetile);
                    tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
                    reconfig_data_format_srca(cb_in0);
#endif
                    copy_init(cb_in0);
                    copy_tile(cb_in0, 0, dst0);

                    if (do_mask_h && last_row) {
#if defined FP32_DEST_ACC_EN
                        reconfig_data_format_srca(cb_mask_h_w);
#endif
                        copy_init(cb_mask_h_w);
                        copy_tile(cb_mask_h_w, 0, dst1);
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }

                    if (do_mask_w && last_col) {
#if defined FP32_DEST_ACC_EN
                        reconfig_data_format_srca(cb_mask_h_w);
#endif
                        copy_init(cb_mask_h_w);
                        copy_tile(cb_mask_h_w, 1, dst1);
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    dfb_intermed0_obj.reserve_back(onetile);
#if defined FP32_DEST_ACC_EN
                    pack_reconfig_data_format(cb_intermed0);
#endif
                    pack_tile(dst0, cb_intermed0);
                    dfb_intermed0_obj.push_back(onetile);
                    tile_regs_release();

                    dfb_in0_obj.pop_front(onetile);
                }

                const auto reduce_block = compute_kernel_lib::ReduceInputBlockShape::single();
                const auto reduce_layout = compute_kernel_lib::ReduceInputMemoryLayout::contiguous();
                const auto reduce_accum = compute_kernel_lib::Accumulate::at(cb_intermed1, num_tile_done);
                if (do_mask) {
                    if (last_out) {
                        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_intermed0, cb_scaler, cb_out0>(
                            reduce_block, reduce_layout, reduce_accum);
                    } else {
                        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_intermed0, cb_scaler, cb_intermed1>(
                            reduce_block, reduce_layout, reduce_accum);
                    }
                } else {
                    if (last_out) {
                        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_in0, cb_scaler, cb_out0>(
                            reduce_block, reduce_layout, reduce_accum);
                    } else {
                        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_in0, cb_scaler, cb_intermed1>(
                            reduce_block, reduce_layout, reduce_accum);
                    }
                }

                num_tile_done++;
            }
        }
    }
}
