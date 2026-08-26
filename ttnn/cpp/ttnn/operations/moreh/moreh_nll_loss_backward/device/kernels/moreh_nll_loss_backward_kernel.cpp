// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    DataflowBuffer dfb_output_grad_obj(dfb::output_grad);
    DataflowBuffer dfb_tmp_weight_obj(dfb::tmp_weight);
    DataflowBuffer dfb_input_grad_obj(dfb::input_grad);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    init_sfpu(dfb::output_grad, dfb::input_grad);

    if constexpr (is_null_binding(dfb::divisor)) {
        dfb_output_grad_obj.wait_front(onetile);

        for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
            dfb_tmp_weight_obj.wait_front(onetile);

            dfb_input_grad_obj.reserve_back(onetile);

            tile_regs_acquire();
            mul_bcast_scalar_init_with_dt(dfb_tmp_weight_obj, dfb_output_grad_obj);
            mul_tiles_bcast_scalar(dfb::tmp_weight, dfb::output_grad, 0, 0, dst0);
            negative_tile_init();
            negative_tile(dst0);

            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_input_grad_obj);
            tile_regs_release();

            dfb_input_grad_obj.push_back(onetile);

            dfb_tmp_weight_obj.pop_front(onetile);
        }
    }

    with_nullable_token(dfb::divisor, [&](const DFBBindingToken& divisor_tok) {
        with_nullable_token(dfb::tmp1, [&](const DFBBindingToken& tmp1_tok) {
            with_nullable_token(dfb::tmp2, [&](const DFBBindingToken& tmp2_tok) {
                DataflowBuffer dfb_divisor_obj(divisor_tok);
                DataflowBuffer dfb_tmp1_obj(tmp1_tok);
                DataflowBuffer dfb_tmp2_obj(tmp2_tok);

                dfb_divisor_obj.wait_front(onetile);
                dfb_tmp1_obj.reserve_back(onetile);

                tile_regs_acquire();
                copy_tile_init_with_dt(dfb_divisor_obj);
                copy_tile(divisor_tok, 0, dst0);
                recip_tile_init();
                recip_tile(dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_tmp1_obj);
                tile_regs_release();

                dfb_tmp1_obj.push_back(onetile);

                dfb_output_grad_obj.wait_front(onetile);

                for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
                    dfb_tmp_weight_obj.wait_front(onetile);
                    dfb_tmp2_obj.reserve_back(onetile);

                    tile_regs_acquire();
                    mul_bcast_scalar_init_with_dt(dfb_tmp_weight_obj, dfb_output_grad_obj);
                    mul_tiles_bcast_scalar(dfb::tmp_weight, dfb::output_grad, 0, 0, dst0);
                    negative_tile_init();
                    negative_tile(dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, dfb_tmp2_obj);
                    tile_regs_release();

                    dfb_tmp2_obj.push_back(onetile);
                    dfb_tmp_weight_obj.pop_front(onetile);

                    dfb_input_grad_obj.reserve_back(onetile);
                    dfb_tmp2_obj.wait_front(onetile);
                    dfb_tmp1_obj.wait_front(onetile);

                    tile_regs_acquire();
                    mul_bcast_scalar_init_with_dt(dfb_tmp2_obj, dfb_tmp1_obj);
                    mul_tiles_bcast_scalar(tmp2_tok, tmp1_tok, 0, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, dfb_input_grad_obj);
                    tile_regs_release();

                    dfb_input_grad_obj.push_back(onetile);
                    dfb_tmp2_obj.pop_front(onetile);
                }

                dfb_divisor_obj.pop_front(onetile);
            });
        });
    });
}
