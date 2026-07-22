// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_weight = tt::CBIndex::c_2;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    DataflowBuffer dfb_divisor_obj(cb_divisor);

    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    DataflowBuffer dfb_tmp_weight_obj(cb_tmp_weight);
    constexpr uint32_t cb_tmp_input = tt::CBIndex::c_25;
    DataflowBuffer dfb_tmp_input_obj(cb_tmp_input);
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_26;
    DataflowBuffer dfb_tmp1_obj(cb_tmp1);
    constexpr uint32_t cb_divisor_recip = tt::CBIndex::c_27;
    DataflowBuffer dfb_divisor_recip_obj(cb_divisor_recip);  // 1/divisor
    constexpr uint32_t cb_tmp3 = tt::CBIndex::c_28;
    DataflowBuffer dfb_tmp3_obj(cb_tmp3);

    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    DataflowBuffer dfb_output_obj(cb_output);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(cb_tmp_weight, cb_tmp_input, cb_output);

#if defined(DIVISOR)
    dfb_divisor_obj.wait_front(onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(dfb_divisor_obj);
    copy_tile(cb_divisor, 0, dst0);
    recip_tile_init();
    recip_tile(dst0);
    tile_regs_commit();

    dfb_divisor_obj.pop_front(onetile);
    dfb_divisor_recip_obj.reserve_back(onetile);
    tile_regs_wait();
    pack_tile_with_dt(dst0, dfb_divisor_recip_obj);
    tile_regs_release();
    dfb_divisor_recip_obj.push_back(onetile);
#endif

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        dfb_tmp_input_obj.wait_front(onetile);

        tile_regs_acquire();
        copy_tile_init_with_dt(dfb_tmp_input_obj);
        copy_tile(cb_tmp_input, 0, dst0);

        negative_tile_init();
        negative_tile(dst0);
        tile_regs_commit();

        dfb_tmp_input_obj.pop_front(onetile);

#if defined(WEIGHT)
        dfb_tmp1_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        tile_regs_release();
        dfb_tmp1_obj.push_back(onetile);

        // multiply weight
        dfb_tmp1_obj.wait_front(onetile);
        dfb_tmp_weight_obj.wait_front(onetile);

        tile_regs_acquire();
        mul_tiles_init_with_dt(dfb_tmp1_obj, dfb_tmp_weight_obj);
        mul_tiles(cb_tmp1, cb_tmp_weight, 0, 0, dst0);
        tile_regs_commit();

        dfb_tmp_weight_obj.pop_front(onetile);
        dfb_tmp1_obj.pop_front(onetile);

#if defined(DIVISOR)
        dfb_tmp3_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp3_obj);
        tile_regs_release();
        dfb_tmp3_obj.push_back(onetile);

        dfb_tmp3_obj.wait_front(onetile);
        dfb_divisor_recip_obj.wait_front(onetile);
        tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
        reconfig_data_format(cb_tmp3, cb_divisor_recip);
#endif
        mul_tiles_bcast_scalar_init_short(cb_tmp3, cb_divisor_recip);
        mul_tiles_bcast_scalar(cb_tmp3, cb_divisor_recip, 0, 0, dst0);
        tile_regs_commit();
        dfb_tmp3_obj.pop_front(onetile);

        dfb_output_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_output_obj);
        tile_regs_release();
        dfb_output_obj.push_back(onetile);
#else
        dfb_output_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_output_obj);
        tile_regs_release();
        dfb_output_obj.push_back(onetile);
#endif
#else
#if defined(DIVISOR)
        dfb_tmp1_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        tile_regs_release();
        dfb_tmp1_obj.push_back(onetile);

        dfb_divisor_recip_obj.wait_front(onetile);
        dfb_tmp1_obj.wait_front(onetile);

        tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
        reconfig_data_format(cb_tmp1, cb_divisor_recip);
#endif
        mul_tiles_bcast_scalar_init_short(cb_tmp1, cb_divisor_recip);
        mul_tiles_bcast_scalar(cb_tmp1, cb_divisor_recip, 0, 0, dst0);
        tile_regs_commit();

        dfb_tmp1_obj.pop_front(onetile);

        dfb_output_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_output_obj);
        tile_regs_release();
        dfb_output_obj.push_back(onetile);
#else
        dfb_output_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_output_obj);
        tile_regs_release();
        dfb_output_obj.push_back(onetile);
#endif
#endif
    }

#if defined(DIVISOR)
    dfb_divisor_recip_obj.pop_front(onetile);
#endif
}
