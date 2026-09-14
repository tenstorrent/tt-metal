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

#if defined(DIVISOR)
    DataflowBuffer dfb_divisor_obj(dfb::divisor);
#endif
    DataflowBuffer dfb_output_grad_obj(dfb::output_grad);
    DataflowBuffer dfb_tmp_weight_obj(dfb::tmp_weight);
#if defined(DIVISOR)
    DataflowBuffer dfb_tmp1_obj(dfb::tmp1);
    DataflowBuffer dfb_tmp2_obj(dfb::tmp2);
#endif
    DataflowBuffer dfb_input_grad_obj(dfb::input_grad);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::output_grad, dfb::input_grad);

#if defined(DIVISOR)
    dfb_divisor_obj.wait_front(onetile);
    dfb_tmp1_obj.reserve_back(onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(dfb_divisor_obj);
    copy_tile(dfb::divisor, 0, dst0);
    recip_tile_init();
    recip_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, dfb_tmp1_obj);
    tile_regs_release();

    dfb_tmp1_obj.push_back(onetile);
#endif

    dfb_output_grad_obj.wait_front(onetile);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
#if defined(DIVISOR)
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
        mul_tiles_bcast_scalar(dfb::tmp2, dfb::tmp1, 0, 0, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_input_grad_obj);
        tile_regs_release();

        dfb_input_grad_obj.push_back(onetile);
        dfb_tmp2_obj.pop_front(onetile);

#else
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
#endif
    }

#if defined(DIVISOR)
    dfb_divisor_obj.pop_front(onetile);
#endif
}
