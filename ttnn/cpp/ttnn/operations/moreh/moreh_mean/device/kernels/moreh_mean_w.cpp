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
#include "experimental/kernel_args.h"

void kernel_main() {
    // Carries the per-core work-split count (the host's units_per_core_group_N), not a tile height.
    uint32_t Ht = get_arg(args::units_per_core);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t NC = get_arg(args::NC);
    constexpr uint32_t origin_W = get_arg(args::origin_W);

    // Selected at runtime between the input DFB and the masked-input DFB; stays uint32_t-valued so the
    // reassignment below is legal (DFBAccessor converts to uint32_t at compile time).
    uint32_t cb_input = dfb::input;
    DataflowBuffer dfb_scaler_obj(dfb::scaler);
    DataflowBuffer dfb_mask_w_obj(dfb::mask_w);
    DataflowBuffer dfb_accum_dst_obj(dfb::accum_dst);
    DataflowBuffer dfb_masked_input_obj(dfb::masked_input);
    DataflowBuffer dfb_out_obj(dfb::out);
    constexpr bool do_mask_w = (origin_W % TILE_WIDTH) != 0;

    compute_kernel_hw_startup(cb_input, cb_input, dfb::out);

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
            cb_input = dfb::input;
            bool is_w_single_tile = (Wt == 1);
            if (!is_w_single_tile) {
                tile_regs_acquire();

                for (uint32_t wt = 0; wt < Wt - 1; ++wt) {
                    DataflowBuffer(cb_input).wait_front(onetile);
#if defined FP32_DEST_ACC_EN
                    reconfig_data_format(cb_input, dfb::scaler);
#endif
                    matmul_init(cb_input, dfb::scaler, false);
                    matmul_tiles(cb_input, dfb::scaler, 0, 0, reduce_dst_idx);
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
                copy_tile(dfb::mask_w, 0, mask_dst_idx);

                mask_tile_init();
                mask_tile(reduce_dst_idx, mask_dst_idx);
                tile_regs_commit();

                dfb_masked_input_obj.reserve_back(onetile);
                tile_regs_wait();
                pack_tile_with_dt(reduce_dst_idx, dfb_masked_input_obj);
                tile_regs_release();
                dfb_masked_input_obj.push_back(onetile);

                DataflowBuffer(cb_input).pop_front(onetile);
                cb_input = dfb::masked_input;
            }

            tile_regs_acquire();
            DataflowBuffer(cb_input).wait_front(onetile);
            if (!is_w_single_tile) {
                dfb_accum_dst_obj.wait_front(onetile);

                copy_tile_init_with_dt(dfb_accum_dst_obj);
                copy_tile(dfb::accum_dst, 0, reduce_dst_idx);
            }

#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_input, dfb::scaler);
#endif
            matmul_init(cb_input, dfb::scaler, false);
            matmul_tiles(cb_input, dfb::scaler, 0, 0, reduce_dst_idx);
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
