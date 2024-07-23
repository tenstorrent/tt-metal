// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_weight = tt::CB::c_in2;
    constexpr uint32_t cb_divisor = tt::CB::c_in3;

    constexpr uint32_t cb_tmp_weight = tt::CB::c_intermed0;
    constexpr uint32_t cb_tmp_input = tt::CB::c_intermed1;
    constexpr uint32_t cb_tmp1 = tt::CB::c_intermed2;
    constexpr uint32_t cb_divisor_recip = tt::CB::c_intermed3;  // 1/divisor
    constexpr uint32_t cb_tmp3 = tt::CB::c_intermed4;

    constexpr uint32_t cb_output = tt::CB::c_out0;

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_tmp_weight, cb_tmp_input);

#if defined(DIVISOR)
    cb_wait_front(cb_divisor, onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(cb_divisor);
    copy_tile(cb_divisor, 0, dst0);
    recip_tile_init();
    recip_tile(dst0);
    tile_regs_commit();

    cb_pop_front(cb_divisor, onetile);
    cb_reserve_back(cb_divisor_recip, onetile);
    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_divisor_recip);
    tile_regs_release();
    cb_push_back(cb_divisor_recip, onetile);
#endif

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb_wait_front(cb_tmp_input, onetile);

        tile_regs_acquire();
        copy_tile_init_with_dt(cb_tmp_input);
        copy_tile(cb_tmp_input, 0, dst0);

        negative_tile_init();
        negative_tile(dst0);
        tile_regs_commit();

        cb_pop_front(cb_tmp_input, onetile);

#if defined(WEIGHT)
        cb_reserve_back(cb_tmp1, onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        tile_regs_release();
        cb_push_back(cb_tmp1, onetile);

        // multiply weight
        cb_wait_front(cb_tmp1, onetile);
        cb_wait_front(cb_tmp_weight, onetile);

        tile_regs_acquire();
        mul_tiles_init_with_dt(cb_tmp1, cb_tmp_weight);
        mul_tiles(cb_tmp1, cb_tmp_weight, 0, 0, dst0);
        tile_regs_commit();

        cb_pop_front(cb_tmp_weight, onetile);
        cb_pop_front(cb_tmp1, onetile);

#if defined(DIVISOR)
        cb_reserve_back(cb_tmp3, onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp3);
        tile_regs_release();
        cb_push_back(cb_tmp3, onetile);

        cb_wait_front(cb_tmp3, onetile);
        cb_wait_front(cb_divisor_recip, onetile);
        tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
        unpack_reconfig_data_format(cb_tmp3, cb_divisor_recip);
#endif
        mul_tiles_bcast_scalar_init_short(cb_tmp3, cb_divisor_recip);
        mul_tiles_bcast_scalar(cb_tmp3, cb_divisor_recip, 0, 0, dst0);
        tile_regs_commit();
        cb_pop_front(cb_tmp3, onetile);

        cb_reserve_back(cb_output, onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        tile_regs_release();
        cb_push_back(cb_output, onetile);
#else
        cb_reserve_back(cb_output, onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        tile_regs_release();
        cb_push_back(cb_output, onetile);
#endif
#else
#if defined(DIVISOR)
        cb_reserve_back(cb_tmp1, onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        tile_regs_release();
        cb_push_back(cb_tmp1, onetile);

        cb_wait_front(cb_divisor_recip, onetile);
        cb_wait_front(cb_tmp1, onetile);

        tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
        unpack_reconfig_data_format(cb_tmp1, cb_divisor_recip);
#endif
        mul_tiles_bcast_scalar_init_short();
        mul_tiles_bcast_scalar(cb_tmp1, cb_divisor_recip, 0, 0, dst0);
        tile_regs_commit();

        cb_pop_front(cb_tmp1, onetile);

        cb_reserve_back(cb_output, onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        tile_regs_release();
        cb_push_back(cb_output, onetile);
#else
        cb_reserve_back(cb_output, onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        tile_regs_release();
        cb_push_back(cb_output, onetile);
#endif
#endif
    }

#if defined(DIVISOR)
    cb_pop_front(cb_divisor_recip, onetile);
#endif
}
}  // namespace NAMESPACE
