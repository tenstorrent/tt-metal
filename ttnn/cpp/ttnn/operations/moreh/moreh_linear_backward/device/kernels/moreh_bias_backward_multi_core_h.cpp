// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace {

constexpr auto cb_in0 = tt::CBIndex::c_0;
constexpr auto cb_scaler = tt::CBIndex::c_1;
constexpr auto cb_mask_h_w = tt::CBIndex::c_2;
constexpr auto cb_out0 = tt::CBIndex::c_16;
constexpr auto cb_intermed0 = tt::CBIndex::c_24;
constexpr auto cb_intermed1 = tt::CBIndex::c_25;

constexpr uint32_t onetile = 1;
constexpr uint32_t dst0 = 0;
constexpr uint32_t dst1 = 1;

// Reduce `rows` consecutive tiles of the current column (streamed one tile at a time through
// cb_in0) into a single tile in cb_out. Successive calls chain through cb_intermed1: iteration 0
// seeds the accumulator, later iterations reload it before reducing.
template <uint32_t cb_out>
ALWI void reduce_column_chunk(uint32_t rows, uint32_t iteration, compute_kernel_lib::ReducePartialScaler partial) {
    compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_in0, cb_scaler, cb_out>(
        compute_kernel_lib::ReduceInputBlockShape::col(rows),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::Accumulate::at(cb_intermed1, iteration),
        compute_kernel_lib::NoOp{},
        partial);
}

// Apply the W mask to the already reduced column tile and pack it out.
//
// The W mask is constant along H and mask_tile zeroes lanes instead of scaling them, so masking
// the reduced tile is equivalent to masking every input tile before the reduce - including for
// non-finite padding, which mask_tile overwrites with 0 either way. That turns the W-mask cost
// from one copy/mask/pack per input tile into one per column.
ALWI void mask_w_and_pack(DataflowBuffer& dfb_intermed0, DataflowBuffer& dfb_out0) {
    dfb_intermed0.wait_front(onetile);
    tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format_srca(cb_intermed0);
#endif
    copy_tile_to_dst_init_short(cb_intermed0);
    copy_tile(cb_intermed0, 0, dst0);

#if defined FP32_DEST_ACC_EN
    reconfig_data_format_srca(cb_mask_h_w);
#endif
    copy_tile_to_dst_init_short(cb_mask_h_w);
    copy_tile(cb_mask_h_w, 1, dst1);
    mask_tile_init();
    mask_tile(dst0, dst1);
    tile_regs_commit();

    tile_regs_wait();
    dfb_out0.reserve_back(onetile);
#if defined FP32_DEST_ACC_EN
    pack_reconfig_data_format(cb_out0);
#endif
    pack_tile(dst0, cb_out0);
    dfb_out0.push_back(onetile);
    tile_regs_release();

    dfb_intermed0.pop_front(onetile);
}

}  // namespace

void kernel_main() {
    ArgFetcher arg_fetcher;
    const uint32_t batch_num = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t Ht = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t Wt_per_core = arg_fetcher.get_next_arg_val<uint32_t>();
    const bool do_mask_h = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);
    const bool do_mask_w = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);

    DataflowBuffer dfb_mask_h_w_obj(cb_mask_h_w);
    DataflowBuffer dfb_intermed0_obj(cb_intermed0);
    DataflowBuffer dfb_out0_obj(cb_out0);

    compute_kernel_hw_startup(cb_in0, cb_in0, cb_out0);

    // A ragged H is handled by the partial scaler, so the mask tiles are only read for a ragged W.
    if (do_mask_w) {
        dfb_mask_h_w_obj.wait_front(onetile * 2);
    }

    // The reader walks each column top to bottom, batch after batch, so one column is a single
    // contiguous run of batch_num * Ht tiles in cb_in0.
    const uint32_t tiles_per_column = batch_num * Ht;

    for (uint32_t wt = 0; wt < Wt_per_core; ++wt) {
        const bool mask_column = (do_mask_w && wt == Wt_per_core - 1);

        if (do_mask_h) {
            // The partial scaler applies to the last H tile of each reduce() call and every batch
            // ends with the ragged H tile, so reduce a batch at a time and accumulate across them.
            const auto partial = compute_kernel_lib::ReducePartialScaler::with_partial();
            for (uint32_t b = 0; b < batch_num; ++b) {
                if (b + 1 < batch_num) {
                    reduce_column_chunk<cb_intermed1>(Ht, b, partial);
                } else if (mask_column) {
                    reduce_column_chunk<cb_intermed0>(Ht, b, partial);
                } else {
                    reduce_column_chunk<cb_out0>(Ht, b, partial);
                }
            }
        } else {
            // H is tile aligned: the whole column is one reduce, no accumulator round trip.
            const auto partial = compute_kernel_lib::ReducePartialScaler::none();
            if (mask_column) {
                reduce_column_chunk<cb_intermed0>(tiles_per_column, 0, partial);
            } else {
                reduce_column_chunk<cb_out0>(tiles_per_column, 0, partial);
            }
        }

        if (mask_column) {
            mask_w_and_pack(dfb_intermed0_obj, dfb_out0_obj);
        }
    }
}
