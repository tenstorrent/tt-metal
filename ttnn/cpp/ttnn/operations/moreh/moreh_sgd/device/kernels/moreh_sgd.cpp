// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr auto dfb_param_in_id = tt::CBIndex::c_0;
    constexpr auto dfb_grad_id = tt::CBIndex::c_1;
    constexpr auto dfb_momentum_in_id = tt::CBIndex::c_2;

    constexpr auto dfb_param_out_id = tt::CBIndex::c_16;
    constexpr auto dfb_momentum_out_id = tt::CBIndex::c_17;

    constexpr auto dfb_scalar_args_id = tt::CBIndex::c_24;
    DataflowBuffer dfb_scalar_args_obj(dfb_scalar_args_id);
    constexpr auto dfb_tmp1_id = tt::CBIndex::c_25;
    constexpr auto dfb_tmp2_id = tt::CBIndex::c_26;
    constexpr auto dfb_tmp3_id = tt::CBIndex::c_27;
    constexpr auto dfb_tmp4_id = tt::CBIndex::c_28;

    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t momentum_tile = 1;
    constexpr uint32_t dampening_tile = 2;
    constexpr uint32_t weight_decay_tile = 3;
    constexpr uint32_t one_tile = 4;

    compute_kernel_hw_startup(dfb_param_in_id, dfb_param_in_id, dfb_param_out_id);

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    // from reader
    dfb_scalar_args_obj.wait_front(5);

#if defined(WEIGHT_DECAY)
    constexpr auto dfb_grad_tmp_id = dfb_tmp2_id;
#else
    constexpr auto dfb_grad_tmp_id = dfb_grad_id;
#endif

#if defined(MOMENTUM)
#if defined(MOMENTUM_INITIALIZED)
    constexpr auto dfb_momentum_tmp_id = dfb_tmp1_id;
#else
    constexpr auto dfb_momentum_tmp_id = dfb_grad_tmp_id;
#endif
#if defined(NESTEROV)
    constexpr auto dfb_final_grad_id = dfb_tmp4_id;
#else
    constexpr auto dfb_final_grad_id = dfb_momentum_tmp_id;
#endif
#else
    constexpr auto dfb_final_grad_id = dfb_grad_tmp_id;
#endif

    for (uint32_t n = 0; n < num_tiles; ++n) {
#if defined(WEIGHT_DECAY)
        // grad += param * weight_decay
        mul_tiles_to_dfb<dfb_param_in_id, dfb_scalar_args_id, dfb_tmp1_id>(
            0, weight_decay_tile, /*pop0=*/0, /*pop1=*/0);

        add_tiles_to_dfb<dfb_grad_id, dfb_tmp1_id, dfb_tmp2_id>();
#endif  // WEIGHT_DECAY

#if defined(MOMENTUM)
#if defined(MOMENTUM_INITIALIZED)
        // grad * (1 - dampening)
        sub_tiles_to_dfb<dfb_scalar_args_id, dfb_scalar_args_id, dfb_tmp1_id>(
            one_tile, dampening_tile, /*pop0=*/0, /*pop1=*/0);

        mul_tiles_to_dfb<dfb_grad_tmp_id, dfb_tmp1_id, dfb_tmp3_id>(0, 0, /*pop0=*/0);

        // momentum_v * momentum
        mul_tiles_to_dfb<dfb_momentum_in_id, dfb_scalar_args_id, dfb_tmp4_id>(0, momentum_tile, /*pop0=*/1, /*pop1=*/0);

        add_tiles_to_dfb<dfb_tmp3_id, dfb_tmp4_id, dfb_tmp1_id>();
#endif

        copy_tile_to_dfb<dfb_momentum_tmp_id, dfb_momentum_out_id>(0, /*pop=*/0);

#if defined(NESTEROV)
        // grad = grad + momentum_v * momentum
        constexpr uint32_t pop_momentum = dfb_grad_tmp_id != dfb_momentum_tmp_id;
        mul_tiles_to_dfb<dfb_momentum_tmp_id, dfb_scalar_args_id, dfb_tmp3_id>(
            0, momentum_tile, /*pop0=*/pop_momentum, /*pop1=*/0);

        add_tiles_to_dfb<dfb_tmp3_id, dfb_grad_tmp_id, dfb_tmp4_id>();
#else
// have to pop dfb_grad_tmp_id
#if defined(MOMENTUM_INITIALIZED)
        DataflowBuffer dfb_grad_tmp_obj(dfb_grad_tmp_id);
        dfb_grad_tmp_obj.pop_front(1);
#else
// not pop this case because `dfb_momentum_tmp_id == dfb_grad_tmp_id`
#endif
#endif

#endif  // MOMENTUM

        // param_out = param_in - lr * grad
        mul_tiles_to_dfb<dfb_scalar_args_id, dfb_final_grad_id, dfb_tmp3_id>(lr_tile, 0, /*pop0=*/0);

        sub_tiles_to_dfb<dfb_param_in_id, dfb_tmp3_id, dfb_param_out_id>();
    }
}
