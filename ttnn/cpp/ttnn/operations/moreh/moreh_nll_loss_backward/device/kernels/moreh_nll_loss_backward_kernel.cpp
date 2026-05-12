// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    const uint32_t tile_offset = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    experimental::CircularBuffer cb_divisor_obj(cb_divisor);
    constexpr uint32_t cb_output_grad = tt::CBIndex::c_0;
    experimental::CircularBuffer cb_output_grad_obj(cb_output_grad);
    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    experimental::CircularBuffer cb_tmp_weight_obj(cb_tmp_weight);
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
    experimental::CircularBuffer cb_tmp1_obj(cb_tmp1);
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
    experimental::CircularBuffer cb_tmp2_obj(cb_tmp2);
    constexpr uint32_t cb_input_grad = tt::CBIndex::c_16;
    experimental::CircularBuffer cb_input_grad_obj(cb_input_grad);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    init_sfpu(cb_output_grad, tt::CBIndex::c_16);

#if defined(DIVISOR)
    cb_divisor_obj.wait_front(onetile);
    cb_tmp1_obj.reserve_back(onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(cb_divisor);
    copy_tile(cb_divisor, 0, dst0);
    recip_tile_init();
    recip_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_tmp1);
    tile_regs_release();

    cb_tmp1_obj.push_back(onetile);
#endif

    cb_output_grad_obj.wait_front(onetile);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
#if defined(DIVISOR)
        cb_tmp_weight_obj.wait_front(onetile);
        cb_tmp2_obj.reserve_back(onetile);

        tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short_with_dt(cb_tmp_weight, cb_output_grad);
        mul_tiles_bcast_scalar(cb_tmp_weight, cb_output_grad, 0, 0, dst0);
        negative_tile_init();
        negative_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp2);
        tile_regs_release();

        cb_tmp2_obj.push_back(onetile);
        cb_tmp_weight_obj.pop_front(onetile);

        cb_input_grad_obj.reserve_back(onetile);
        cb_tmp2_obj.wait_front(onetile);
        cb_tmp1_obj.wait_front(onetile);

        tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short_with_dt(cb_tmp2, cb_tmp1);
        mul_tiles_bcast_scalar(cb_tmp2, cb_tmp1, 0, 0, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_input_grad);
        tile_regs_release();

        cb_input_grad_obj.push_back(onetile);
        cb_tmp2_obj.pop_front(onetile);

#else
        cb_tmp_weight_obj.wait_front(onetile);

        cb_input_grad_obj.reserve_back(onetile);

        tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short_with_dt(cb_tmp_weight, cb_output_grad);
        mul_tiles_bcast_scalar(cb_tmp_weight, cb_output_grad, 0, 0, dst0);
        negative_tile_init();
        negative_tile(dst0);

        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_input_grad);
        tile_regs_release();

        cb_input_grad_obj.push_back(onetile);

        cb_tmp_weight_obj.pop_front(onetile);
#endif
    }

#if defined(DIVISOR)
    cb_divisor_obj.pop_front(onetile);
#endif
}
