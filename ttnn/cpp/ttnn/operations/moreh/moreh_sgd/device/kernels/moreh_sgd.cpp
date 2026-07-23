// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr auto cb_param_in = tt::CBIndex::c_0;
    constexpr auto cb_grad = tt::CBIndex::c_1;
    constexpr auto cb_momentum_in = tt::CBIndex::c_2;

    constexpr auto cb_param_out = tt::CBIndex::c_16;
    constexpr auto cb_momentum_out = tt::CBIndex::c_17;

    constexpr auto cb_scalar_args = tt::CBIndex::c_24;
    DataflowBuffer dfb_scalar_args_obj(cb_scalar_args);
    constexpr auto cb_tmp1 = tt::CBIndex::c_25;
    constexpr auto cb_tmp2 = tt::CBIndex::c_26;
    constexpr auto cb_tmp3 = tt::CBIndex::c_27;
    constexpr auto cb_tmp4 = tt::CBIndex::c_28;

    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t momentum_tile = 1;
    constexpr uint32_t dampening_tile = 2;
    constexpr uint32_t weight_decay_tile = 3;
    constexpr uint32_t one_tile = 4;

    binary_op_init_common(cb_param_in, cb_param_in, cb_param_out);

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    // from reader
    dfb_scalar_args_obj.wait_front(5);

#if defined(WEIGHT_DECAY)
    constexpr auto cb_grad_tmp = cb_tmp2;
#else
    constexpr auto cb_grad_tmp = cb_grad;
#endif

#if defined(MOMENTUM)
#if defined(MOMENTUM_INITIALIZED)
    constexpr auto cb_momentum_tmp = cb_tmp1;
#else
    constexpr auto cb_momentum_tmp = cb_grad_tmp;
#endif
#if defined(NESTEROV)
    constexpr auto cb_final_grad = cb_tmp4;
#else
    constexpr auto cb_final_grad = cb_momentum_tmp;
#endif
#else
    constexpr auto cb_final_grad = cb_grad_tmp;
#endif

    for (uint32_t n = 0; n < num_tiles; ++n) {
#if defined(WEIGHT_DECAY)
        // grad += param * weight_decay
        mul_tiles_to_cb<cb_param_in, cb_scalar_args, cb_tmp1>(0, weight_decay_tile, /*pop0=*/0, /*pop1=*/0);

        add_tiles_to_cb<cb_grad, cb_tmp1, cb_tmp2>();
#endif  // WEIGHT_DECAY

#if defined(MOMENTUM)
#if defined(MOMENTUM_INITIALIZED)
        // grad * (1 - dampening)
        sub_tiles_to_cb<cb_scalar_args, cb_scalar_args, cb_tmp1>(one_tile, dampening_tile, /*pop0=*/0, /*pop1=*/0);

        mul_tiles_to_cb<cb_grad_tmp, cb_tmp1, cb_tmp3>(0, 0, /*pop0=*/0);

        // momentum_v * momentum
        mul_tiles_to_cb<cb_momentum_in, cb_scalar_args, cb_tmp4>(0, momentum_tile, /*pop0=*/1, /*pop1=*/0);

        add_tiles_to_cb<cb_tmp3, cb_tmp4, cb_tmp1>();
#endif

        copy_tile_to_cb<cb_momentum_tmp, cb_momentum_out>(0, /*pop=*/0);

#if defined(NESTEROV)
        // grad = grad + momentum_v * momentum
        constexpr uint32_t pop_momentum = cb_grad_tmp != cb_momentum_tmp;
        mul_tiles_to_cb<cb_momentum_tmp, cb_scalar_args, cb_tmp3>(0, momentum_tile, /*pop0=*/pop_momentum, /*pop1=*/0);

        add_tiles_to_cb<cb_tmp3, cb_grad_tmp, cb_tmp4>();
#else
// have to pop cb_grad_tmp
#if defined(MOMENTUM_INITIALIZED)
        cb_pop_front(cb_grad_tmp, 1);
#else
// not pop this case because `cb_momentum_tmp == cb_grad_tmp`
#endif
#endif

#endif  // MOMENTUM

        // param_out = param_in - lr * grad
        mul_tiles_to_cb<cb_scalar_args, cb_final_grad, cb_tmp3>(lr_tile, 0, /*pop0=*/0);

        sub_tiles_to_cb<cb_param_in, cb_tmp3, cb_param_out>();
    }
}
