// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    constexpr int onetile = 1;
    const uint32_t batch_num = get_arg(args::batch_num);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Wt = get_arg(args::Wt);
    const bool do_mask_h = (get_arg(args::do_mask_h) == 1);
    const bool do_mask_w = (get_arg(args::do_mask_w) == 1);

    // in0 holds the output_grad tiles the reader streams in; scaler and mask_h_w are the reader's
    // prepared constant tiles; intermed0 stages a masked input tile; intermed1 carries the running
    // reduction accumulator; out is the bias_grad tile the writer drains.
    DataflowBuffer dfb_in0_obj(dfb::in0);
    DataflowBuffer dfb_scaler_obj(dfb::scaler);
    // The mask buffer is only allocated when a mask applies, so the host binds it — and defines
    // DO_MASK_H_W — on exactly that condition. Without the binding there is no dfb::mask_h_w token
    // to name, so every reference to it sits behind the preprocessor gate. The runtime checks below
    // are left exactly as they were: they coincide with the define when it is set, and rewriting
    // them would be a change to the kernel's logic rather than to its bindings.
#ifdef DO_MASK_H_W
    DataflowBuffer dfb_mask_h_w_obj(dfb::mask_h_w);
#endif
    DataflowBuffer dfb_intermed0_obj(dfb::intermed0);
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    compute_kernel_hw_startup(dfb::in0, dfb::in0, dfb::out);
    dfb_scaler_obj.wait_front(onetile);

#ifdef DO_MASK_H_W
    if (do_mask_h || do_mask_w) {
        dfb_mask_h_w_obj.wait_front(onetile * 2);
    }
#endif

    uint32_t num_tiles = batch_num * Ht * Wt;
    uint32_t num_tile_done = 0;
    for (uint32_t b = 0; b < batch_num; ++b) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                bool last_row = (ht == Ht - 1);
                bool last_col = (wt == Wt - 1);
                bool last_out = (num_tile_done == num_tiles - 1);
                bool do_mask = (do_mask_h && last_row) || (do_mask_w && last_col);

                if (do_mask) {
                    // get tile from reader and apply mask
                    dfb_in0_obj.wait_front(onetile);
                    tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
                    reconfig_data_format_srca(dfb::in0);
#endif
                    copy_init(dfb::in0);
                    copy_tile(dfb::in0, 0, dst0);

#ifdef DO_MASK_H_W
                    if (do_mask_h && last_row) {
#if defined FP32_DEST_ACC_EN
                        reconfig_data_format_srca(dfb::mask_h_w);
#endif
                        copy_init(dfb::mask_h_w);
                        copy_tile(dfb::mask_h_w, 0, dst1);
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }

                    if (do_mask_w && last_col) {
#if defined FP32_DEST_ACC_EN
                        reconfig_data_format_srca(dfb::mask_h_w);
#endif
                        copy_init(dfb::mask_h_w);
                        copy_tile(dfb::mask_h_w, 1, dst1);
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }
#endif
                    tile_regs_commit();

                    tile_regs_wait();
                    dfb_intermed0_obj.reserve_back(onetile);
#if defined FP32_DEST_ACC_EN
                    pack_reconfig_data_format(dfb::intermed0);
#endif
                    pack_tile(dst0, dfb::intermed0);
                    dfb_intermed0_obj.push_back(onetile);
                    tile_regs_release();

                    dfb_in0_obj.pop_front(onetile);
                }

                const auto reduce_block = compute_kernel_lib::ReduceInputBlockShape::single();
                const auto reduce_layout = compute_kernel_lib::ReduceInputMemoryLayout::contiguous();
                const auto reduce_accum = compute_kernel_lib::Accumulate::at(dfb::intermed1, num_tile_done);
                if (do_mask) {
                    if (last_out) {
                        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::intermed0, dfb::scaler, dfb::out>(
                            reduce_block, reduce_layout, reduce_accum);
                    } else {
                        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::intermed0, dfb::scaler, dfb::intermed1>(
                            reduce_block, reduce_layout, reduce_accum);
                    }
                } else {
                    if (last_out) {
                        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::in0, dfb::scaler, dfb::out>(
                            reduce_block, reduce_layout, reduce_accum);
                    } else {
                        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::in0, dfb::scaler, dfb::intermed1>(
                            reduce_block, reduce_layout, reduce_accum);
                    }
                }

                num_tile_done++;
            }
        }
    }
}
