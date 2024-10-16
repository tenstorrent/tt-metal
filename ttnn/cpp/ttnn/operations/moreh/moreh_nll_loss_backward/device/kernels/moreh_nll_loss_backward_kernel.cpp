// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    const uint32_t tile_offset = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_divisor = tt::CB::c_in3;
    constexpr uint32_t cb_output_grad = tt::CB::c_in0;
    constexpr uint32_t cb_tmp_weight = tt::CB::c_intermed0;
    constexpr uint32_t cb_tmp1 = tt::CB::c_intermed1;
    constexpr uint32_t cb_tmp2 = tt::CB::c_intermed2;
    constexpr uint32_t cb_input_grad = tt::CB::c_out0;

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    init_sfpu(cb_output_grad);

#if defined(DIVISOR)
    cb_wait_front(cb_divisor, onetile);
    cb_reserve_back(cb_tmp1, onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(cb_divisor);
    copy_tile(cb_divisor, 0, dst0);
    recip_tile_init();
    recip_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_tmp1);
    tile_regs_release();

    cb_push_back(cb_tmp1, onetile);
#endif

    cb_wait_front(cb_output_grad, onetile);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
#if defined(DIVISOR)
        cb_wait_front(cb_tmp_weight, onetile);
        cb_reserve_back(cb_tmp2, onetile);

        tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short_with_dt(cb_tmp_weight, cb_output_grad);
        mul_tiles_bcast_scalar(cb_tmp_weight, cb_output_grad, 0, 0, dst0);
        negative_tile_init();
        negative_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp2);
        tile_regs_release();

        cb_push_back(cb_tmp2, onetile);
        cb_pop_front(cb_tmp_weight, onetile);

        cb_reserve_back(cb_input_grad, onetile);
        cb_wait_front(cb_tmp2, onetile);
        cb_wait_front(cb_tmp1, onetile);

        tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short_with_dt(cb_tmp2, cb_tmp1);
        mul_tiles_bcast_scalar(cb_tmp2, cb_tmp1, 0, 0, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_input_grad);
        tile_regs_release();

        cb_push_back(cb_input_grad, onetile);
        cb_pop_front(cb_tmp2, onetile);

#else
        cb_wait_front(cb_tmp_weight, onetile);

        cb_reserve_back(cb_input_grad, onetile);

        tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short_with_dt(cb_tmp_weight, cb_output_grad);
        mul_tiles_bcast_scalar(cb_tmp_weight, cb_output_grad, 0, 0, dst0);
        negative_tile_init();
        negative_tile(dst0);

        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_input_grad);
        tile_regs_release();

        cb_push_back(cb_input_grad, onetile);

        cb_pop_front(cb_tmp_weight, onetile);
#endif
    }

#if defined(DIVISOR)
    cb_pop_front(cb_divisor, onetile);
#endif
}
}  // namespace NAMESPACE
