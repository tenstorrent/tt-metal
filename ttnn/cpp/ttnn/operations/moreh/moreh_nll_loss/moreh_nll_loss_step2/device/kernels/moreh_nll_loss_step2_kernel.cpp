// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    DataflowBuffer dfb_tmp_weight_obj(dfb::tmp_weight);
    DataflowBuffer dfb_tmp_input_obj(dfb::tmp_input);
    DataflowBuffer dfb_tmp1_obj(dfb::tmp1);
    DataflowBuffer dfb_divisor_recip_obj(dfb::divisor_recip);  // 1/divisor
    DataflowBuffer dfb_tmp3_obj(dfb::tmp3);

    DataflowBuffer dfb_output_obj(dfb::output);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::tmp_weight, dfb::tmp_input, dfb::output);

#if defined(DIVISOR)
    // The divisor buffer only exists when a divisor tensor was supplied, so both the accessor and
    // everything that touches it are gated on the same condition as the host-side binding.
    DataflowBuffer dfb_divisor_obj(dfb::divisor);

    dfb_divisor_obj.wait_front(onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(dfb_divisor_obj);
    copy_tile(dfb::divisor, 0, dst0);
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
        copy_tile(dfb::tmp_input, 0, dst0);

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
        mul_tiles(dfb::tmp1, dfb::tmp_weight, 0, 0, dst0);
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
        reconfig_data_format(dfb::tmp3, dfb::divisor_recip);
#endif
        mul_bcast_scalar_init(dfb::tmp3, dfb::divisor_recip);
        mul_tiles_bcast_scalar(dfb::tmp3, dfb::divisor_recip, 0, 0, dst0);
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
        reconfig_data_format(dfb::tmp1, dfb::divisor_recip);
#endif
        mul_bcast_scalar_init(dfb::tmp1, dfb::divisor_recip);
        mul_tiles_bcast_scalar(dfb::tmp1, dfb::divisor_recip, 0, 0, dst0);
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
