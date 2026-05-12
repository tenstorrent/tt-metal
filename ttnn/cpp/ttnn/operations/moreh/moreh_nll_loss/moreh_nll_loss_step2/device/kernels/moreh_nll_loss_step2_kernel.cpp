// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_weight = tt::CBIndex::c_2;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    experimental::CircularBuffer cb_divisor_obj(cb_divisor);

    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    experimental::CircularBuffer cb_tmp_weight_obj(cb_tmp_weight);
    constexpr uint32_t cb_tmp_input = tt::CBIndex::c_25;
    experimental::CircularBuffer cb_tmp_input_obj(cb_tmp_input);
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_26;
    experimental::CircularBuffer cb_tmp1_obj(cb_tmp1);
    constexpr uint32_t cb_divisor_recip = tt::CBIndex::c_27;
    experimental::CircularBuffer cb_divisor_recip_obj(cb_divisor_recip);  // 1/divisor
    constexpr uint32_t cb_tmp3 = tt::CBIndex::c_28;
    experimental::CircularBuffer cb_tmp3_obj(cb_tmp3);

    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    experimental::CircularBuffer cb_output_obj(cb_output);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_tmp_weight, cb_tmp_input, cb_output);

#if defined(DIVISOR)
    cb_divisor_obj.wait_front(onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(cb_divisor);
    copy_tile(cb_divisor, 0, dst0);
    recip_tile_init();
    recip_tile(dst0);
    tile_regs_commit();

    cb_divisor_obj.pop_front(onetile);
    cb_divisor_recip_obj.reserve_back(onetile);
    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_divisor_recip);
    tile_regs_release();
    cb_divisor_recip_obj.push_back(onetile);
#endif

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb_tmp_input_obj.wait_front(onetile);

        tile_regs_acquire();
        copy_tile_init_with_dt(cb_tmp_input);
        copy_tile(cb_tmp_input, 0, dst0);

        negative_tile_init();
        negative_tile(dst0);
        tile_regs_commit();

        cb_tmp_input_obj.pop_front(onetile);

#if defined(WEIGHT)
        cb_tmp1_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        tile_regs_release();
        cb_tmp1_obj.push_back(onetile);

        // multiply weight
        cb_tmp1_obj.wait_front(onetile);
        cb_tmp_weight_obj.wait_front(onetile);

        tile_regs_acquire();
        mul_tiles_init_with_dt(cb_tmp1, cb_tmp_weight);
        mul_tiles(cb_tmp1, cb_tmp_weight, 0, 0, dst0);
        tile_regs_commit();

        cb_tmp_weight_obj.pop_front(onetile);
        cb_tmp1_obj.pop_front(onetile);

#if defined(DIVISOR)
        cb_tmp3_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp3);
        tile_regs_release();
        cb_tmp3_obj.push_back(onetile);

        cb_tmp3_obj.wait_front(onetile);
        cb_divisor_recip_obj.wait_front(onetile);
        tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
        reconfig_data_format(cb_tmp3, cb_divisor_recip);
#endif
        mul_tiles_bcast_scalar_init_short(cb_tmp3, cb_divisor_recip);
        mul_tiles_bcast_scalar(cb_tmp3, cb_divisor_recip, 0, 0, dst0);
        tile_regs_commit();
        cb_tmp3_obj.pop_front(onetile);

        cb_output_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        tile_regs_release();
        cb_output_obj.push_back(onetile);
#else
        cb_output_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        tile_regs_release();
        cb_output_obj.push_back(onetile);
#endif
#else
#if defined(DIVISOR)
        cb_tmp1_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        tile_regs_release();
        cb_tmp1_obj.push_back(onetile);

        cb_divisor_recip_obj.wait_front(onetile);
        cb_tmp1_obj.wait_front(onetile);

        tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
        reconfig_data_format(cb_tmp1, cb_divisor_recip);
#endif
        mul_tiles_bcast_scalar_init_short(cb_tmp1, cb_divisor_recip);
        mul_tiles_bcast_scalar(cb_tmp1, cb_divisor_recip, 0, 0, dst0);
        tile_regs_commit();

        cb_tmp1_obj.pop_front(onetile);

        cb_output_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        tile_regs_release();
        cb_output_obj.push_back(onetile);
#else
        cb_output_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        tile_regs_release();
        cb_output_obj.push_back(onetile);
#endif
#endif
    }

#if defined(DIVISOR)
    cb_divisor_recip_obj.pop_front(onetile);
#endif
}
