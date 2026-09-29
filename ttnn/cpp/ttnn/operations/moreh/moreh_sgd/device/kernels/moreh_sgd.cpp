// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr auto cb_param_in = tt::CBIndex::c_0;
    DataflowBuffer dfb_param_in_obj(cb_param_in);
    constexpr auto cb_grad = tt::CBIndex::c_1;
    DataflowBuffer dfb_grad_obj(cb_grad);
    constexpr auto cb_momentum_in = tt::CBIndex::c_2;
    DataflowBuffer dfb_momentum_in_obj(cb_momentum_in);

    constexpr auto cb_param_out = tt::CBIndex::c_16;
    DataflowBuffer dfb_param_out_obj(cb_param_out);
    constexpr auto cb_momentum_out = tt::CBIndex::c_17;
    DataflowBuffer dfb_momentum_out_obj(cb_momentum_out);

    constexpr auto cb_scalar_args = tt::CBIndex::c_24;
    DataflowBuffer dfb_scalar_args_obj(cb_scalar_args);
    constexpr auto cb_tmp1 = tt::CBIndex::c_25;
    DataflowBuffer dfb_tmp1_obj(cb_tmp1);
    constexpr auto cb_tmp2 = tt::CBIndex::c_26;
    DataflowBuffer dfb_tmp2_obj(cb_tmp2);
    constexpr auto cb_tmp3 = tt::CBIndex::c_27;
    DataflowBuffer dfb_tmp3_obj(cb_tmp3);
    constexpr auto cb_tmp4 = tt::CBIndex::c_28;
    DataflowBuffer dfb_tmp4_obj(cb_tmp4);

    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t momentum_tile = 1;
    constexpr uint32_t dampening_tile = 2;
    constexpr uint32_t weight_decay_tile = 3;
    constexpr uint32_t one_tile = 4;

    compute_kernel_hw_startup(cb_param_in, cb_param_in, cb_param_out);

    uint32_t num_tiles = get_compile_time_arg_val(0);

    // from reader
    dfb_scalar_args_obj.wait_front(5);

    for (uint32_t n = 0; n < num_tiles; ++n) {
        uint32_t cb_grad_tmp = cb_grad;
#if defined(WEIGHT_DECAY)
        // grad += param * weight_decay
        mul_tiles_to_cb(
            dfb_param_in_obj, dfb_scalar_args_obj, dfb_tmp1_obj, 0, weight_decay_tile, /*pop0=*/0, /*pop1=*/0);

        add_tiles_to_cb(dfb_grad_obj, dfb_tmp1_obj, dfb_tmp2_obj, 0, 0, /*pop0=*/1, /*pop1=*/1);

        cb_grad_tmp = cb_tmp2;
#endif  // WEIGHT_DECAY

#if defined(MOMENTUM)
        uint32_t cb_momentum_tmp = cb_grad_tmp;
#if defined(MOMENTUM_INITIALIZED)
        // grad * (1 - dampening)
        sub_tiles_to_cb(
            dfb_scalar_args_obj, dfb_scalar_args_obj, dfb_tmp1_obj, one_tile, dampening_tile, /*pop0=*/0, /*pop0=*/0);

        {
            DataflowBuffer dfb_grad_tmp_obj(cb_grad_tmp);
            mul_tiles_to_cb(dfb_grad_tmp_obj, dfb_tmp1_obj, dfb_tmp3_obj, 0, 0, /*pop0=*/0, /*pop0=*/1);
        }

        // momentum_v * momentum
        mul_tiles_to_cb(
            dfb_momentum_in_obj, dfb_scalar_args_obj, dfb_tmp4_obj, 0, momentum_tile, /*pop0=*/1, /*pop0=*/0);

        add_tiles_to_cb(dfb_tmp3_obj, dfb_tmp4_obj, dfb_tmp1_obj, 0, 0, /*pop0=*/1, /*pop1=*/1);

        cb_momentum_tmp = cb_tmp1;
#endif

        {
            DataflowBuffer dfb_momentum_tmp_obj(cb_momentum_tmp);
            copy_tile_to_cb(dfb_momentum_tmp_obj, dfb_momentum_out_obj, 0, /*pop=*/0);
        }

#if defined(NESTEROV)
        // grad = grad + momentum_v * momentum
        uint32_t pop_momentum = (cb_grad_tmp != cb_momentum_tmp);
        {
            DataflowBuffer dfb_momentum_tmp_obj(cb_momentum_tmp);
            mul_tiles_to_cb(
                dfb_momentum_tmp_obj,
                dfb_scalar_args_obj,
                dfb_tmp3_obj,
                0,
                momentum_tile,
                /*pop0=*/pop_momentum,
                /*pop1=*/0);
        }

        {
            DataflowBuffer dfb_grad_tmp_obj(cb_grad_tmp);
            add_tiles_to_cb(dfb_tmp3_obj, dfb_grad_tmp_obj, dfb_tmp4_obj, 0, 0, /*pop0=*/1, /*pop1=*/1);
        }

        cb_grad_tmp = cb_tmp4;
#else
// have to pop cb_grad_tmp
#if defined(MOMENTUM_INITIALIZED)
        DataflowBuffer dfb_grad_tmp_obj(cb_grad_tmp);
        dfb_grad_tmp_obj.pop_front(1);
#else
// not pop this case because `cb_momentum_tmp == cb_grad_tmp`
#endif

        cb_grad_tmp = cb_momentum_tmp;
#endif

#endif  // MOMENTUM

        // param_out = param_in - lr * grad
        {
            DataflowBuffer dfb_grad_tmp_obj(cb_grad_tmp);
            mul_tiles_to_cb(dfb_scalar_args_obj, dfb_grad_tmp_obj, dfb_tmp3_obj, lr_tile, 0, /*pop0=*/0, /*pop1=*/1);
        }

        sub_tiles_to_cb(dfb_param_in_obj, dfb_tmp3_obj, dfb_param_out_obj, 0, 0, /*pop0=*/1, /*pop1=*/1);
    }
}
