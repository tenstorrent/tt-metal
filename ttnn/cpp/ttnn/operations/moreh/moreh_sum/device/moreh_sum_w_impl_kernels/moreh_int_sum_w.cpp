// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/sfpu_int_sum.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Carries the per-core work-split count (the host's num_rows_per_core_group_N).
    constexpr uint32_t num_rows = get_arg(args::num_rows);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t origin_W = get_arg(args::origin_W);

    DataflowBuffer dfb_in0_obj(dfb::input);
    DataflowBuffer dfb_mask_w_obj(dfb::mask_w);
    DataflowBuffer dfb_intermed0_obj(dfb::intermed0);
    DataflowBuffer dfb_out0_obj(dfb::out);
    constexpr uint32_t TILE_W = 32;
    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;
    constexpr int onetile = 1;
    constexpr int idx0 = 0;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;

    unary_op_init_common(dfb::input, dfb::out);

    if (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
    }

    for (uint32_t row = 0; row < num_rows; ++row) {
        constexpr bool is_single_wt = (Wt == 1);
        if (is_single_wt) {
            tile_regs_acquire();
            copy_tile_to_dst(dfb_in0_obj, idx0, dst0);

            if (do_mask_w) {
                copy_tile_to_dst(dfb_mask_w_obj, idx0, dst1, false);
                mask_tile_init();
                mask_tile(dst0, dst1, DataFormat::Int32);
            }

            sfpu_sum_int_init();
            sfpu_sum_int_row(dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_from_dst(dfb_out0_obj, dst0);
            tile_regs_release();
        } else {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                if (wt == 0) {
                    tile_regs_acquire();
                    copy_tile_to_dst(dfb_in0_obj, idx0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_from_dst(dfb_intermed0_obj, dst0);
                    tile_regs_release();
                } else {
                    tile_regs_acquire();
                    copy_tile_to_dst(dfb_in0_obj, idx0, dst0);
                    if (wt == Wt - 1 && do_mask_w) {
                        copy_tile_to_dst(dfb_mask_w_obj, idx0, dst1, false);
                        mask_tile_init();
                        mask_tile(dst0, dst1, DataFormat::Int32);
                    }

                    copy_tile_to_dst(dfb_intermed0_obj, idx0, dst1);
                    sfpu_sum_int_init();
                    sfpu_add_int(dst0, dst1);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_from_dst(dfb_intermed0_obj, dst0);
                    tile_regs_release();
                }
            }

            tile_regs_acquire();
            copy_tile_to_dst(dfb_intermed0_obj, idx0, dst0);
            sfpu_sum_int_init();
            sfpu_sum_int_row(dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_from_dst(dfb_out0_obj, dst0);
            tile_regs_release();
        }
    }
}
