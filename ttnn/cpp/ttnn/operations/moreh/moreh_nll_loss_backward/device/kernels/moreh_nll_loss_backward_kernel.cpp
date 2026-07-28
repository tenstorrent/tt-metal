// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    const uint32_t tile_offset = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    DataflowBuffer dfb_divisor_obj(cb_divisor);
    constexpr uint32_t cb_output_grad = tt::CBIndex::c_0;
    DataflowBuffer dfb_output_grad_obj(cb_output_grad);
    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    DataflowBuffer dfb_tmp_weight_obj(cb_tmp_weight);
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
    DataflowBuffer dfb_tmp1_obj(cb_tmp1);
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
    DataflowBuffer dfb_tmp2_obj(cb_tmp2);
    constexpr uint32_t cb_input_grad = tt::CBIndex::c_16;
    DataflowBuffer dfb_input_grad_obj(cb_input_grad);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    init_sfpu(cb_output_grad, tt::CBIndex::c_16);

#if defined(DIVISOR)
    dfb_divisor_obj.wait_front(onetile);
    dfb_tmp1_obj.reserve_back(onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(dfb_divisor_obj);
    copy_tile(cb_divisor, 0, dst0);
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
        mul_tiles_bcast_scalar_init_short_with_dt(dfb_tmp_weight_obj, dfb_output_grad_obj);
        mul_tiles_bcast_scalar(cb_tmp_weight, cb_output_grad, 0, 0, dst0);
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
        mul_tiles_bcast_scalar_init_short_with_dt(dfb_tmp2_obj, dfb_tmp1_obj);
        mul_tiles_bcast_scalar(cb_tmp2, cb_tmp1, 0, 0, dst0);
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
        mul_tiles_bcast_scalar_init_short_with_dt(dfb_tmp_weight_obj, dfb_output_grad_obj);
        mul_tiles_bcast_scalar(cb_tmp_weight, cb_output_grad, 0, 0, dst0);
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
