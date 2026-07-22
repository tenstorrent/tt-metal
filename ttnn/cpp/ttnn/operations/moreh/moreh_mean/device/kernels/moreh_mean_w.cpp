// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/mask.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_W = get_compile_time_arg_val(3);

    auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    DataflowBuffer dfb_scaler_obj(cb_scaler);
    constexpr auto cb_mask_w = tt::CBIndex::c_3;
    DataflowBuffer dfb_mask_w_obj(cb_mask_w);
    constexpr auto cb_accum_dst = tt::CBIndex::c_24;
    DataflowBuffer dfb_accum_dst_obj(cb_accum_dst);
    constexpr auto cb_masked_input = tt::CBIndex::c_25;
    DataflowBuffer dfb_masked_input_obj(cb_masked_input);
    constexpr auto cb_out = tt::CBIndex::c_16;
    DataflowBuffer dfb_out_obj(cb_out);
    constexpr bool do_mask_w = (origin_W % TILE_WIDTH) != 0;

    compute_kernel_hw_startup(cb_input, cb_input, cb_out);

    dfb_scaler_obj.wait_front(1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    const uint32_t mask_dst_idx = reduce_dst_idx + 1;

    if (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            cb_input = tt::CBIndex::c_0;
            bool is_w_single_tile = (Wt == 1);
            if (!is_w_single_tile) {
                tile_regs_acquire();

                for (uint32_t wt = 0; wt < Wt - 1; ++wt) {
                    DataflowBuffer(cb_input).wait_front(onetile);
#if defined FP32_DEST_ACC_EN
                    reconfig_data_format(cb_input, cb_scaler);
#endif
                    matmul_init(cb_input, cb_scaler, false);
                    matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
                    DataflowBuffer(cb_input).pop_front(onetile);
                }
                tile_regs_commit();

                dfb_accum_dst_obj.reserve_back(onetile);
                tile_regs_wait();
                pack_tile_with_dt(reduce_dst_idx, dfb_accum_dst_obj);
                tile_regs_release();
                dfb_accum_dst_obj.push_back(onetile);
            }

            if (do_mask_w) {
                tile_regs_acquire();
                DataflowBuffer(cb_input).wait_front(onetile);

                copy_tile_init_with_dt(DataflowBuffer(cb_input));
                copy_tile(cb_input, 0, reduce_dst_idx);

                copy_tile_init_with_dt(dfb_mask_w_obj);
                copy_tile(cb_mask_w, 0, mask_dst_idx);

                mask_tile_init();
                mask_tile(reduce_dst_idx, mask_dst_idx);
                tile_regs_commit();

                dfb_masked_input_obj.reserve_back(onetile);
                tile_regs_wait();
                pack_tile_with_dt(reduce_dst_idx, dfb_masked_input_obj);
                tile_regs_release();
                dfb_masked_input_obj.push_back(onetile);

                DataflowBuffer(cb_input).pop_front(onetile);
                cb_input = cb_masked_input;
            }

            tile_regs_acquire();
            DataflowBuffer(cb_input).wait_front(onetile);
            if (!is_w_single_tile) {
                dfb_accum_dst_obj.wait_front(onetile);

                copy_tile_init_with_dt(dfb_accum_dst_obj);
                copy_tile(cb_accum_dst, 0, reduce_dst_idx);
            }

#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_input, cb_scaler);
#endif
            matmul_init(cb_input, cb_scaler, false);
            matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
            tile_regs_commit();

            dfb_out_obj.reserve_back(onetile);
            tile_regs_wait();
            pack_tile_with_dt(reduce_dst_idx, dfb_out_obj);
            tile_regs_release();
            dfb_out_obj.push_back(onetile);

            DataflowBuffer(cb_input).pop_front(onetile);
            if (!is_w_single_tile) {
                dfb_accum_dst_obj.pop_front(onetile);
            }
        }
    }

    if (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
    dfb_scaler_obj.pop_front(onetile);
}
