// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

#define ACTIVATION_INIT_RELU relu_tile_init
#define ACTIVATION_APPLY_RELU relu_tile

#define ACTIVATION_INIT_SQUARE square_tile_init
#define ACTIVATION_APPLY_SQUARE square_tile

#define ACTIVATION_INIT_GTZ gtz_tile_init
#define ACTIVATION_APPLY_GTZ gtz_tile

#define ACTIVATION_INIT_LTZ ltz_tile_init
#define ACTIVATION_APPLY_LTZ ltz_tile

#define ACTIVATION_INIT_GEZ gez_tile_init
#define ACTIVATION_APPLY_GEZ gez_tile

#define ACTIVATION_INIT_LEZ lez_tile_init
#define ACTIVATION_APPLY_LEZ lez_tile

#define ACTIVATION_INIT_EQZ eqz_tile_init
#define ACTIVATION_APPLY_EQZ eqz_tile

#define ACTIVATION_INIT_NEZ nez_tile_init
#define ACTIVATION_APPLY_NEZ nez_tile

#define ACTIVATION_INIT_LOG log_tile_init
#define ACTIVATION_APPLY_LOG log_tile

#define ACTIVATION_INIT_LOG2 log_with_base_tile_init
#define ACTIVATION_APPLY_LOG2(i) log_with_base_tile(i, 0x3dc5u)

#define ACTIVATION_INIT_EXP exp_tile_init<false>
#define ACTIVATION_APPLY_EXP exp_tile<false>

#define ACTIVATION_INIT_EXP2 exp2_tile_init
#define ACTIVATION_APPLY_EXP2 exp2_tile

#define ACTIVATION_INIT_RECIP recip_tile_init
#define ACTIVATION_APPLY_RECIP recip_tile

#define ACTIVATION_INIT_GELU gelu_tile_init<false>
#define ACTIVATION_APPLY_GELU gelu_tile<false>

#define IS_EMPTY(...) P_CAT(IS_EMPTY_, IS_BEGIN_PARENS(__VA_ARGS__))(__VA_ARGS__)
#define IS_EMPTY_0(...) IS_BEGIN_PARENS(IS_EMPTY_NON_FUNCTION_C __VA_ARGS__())
#define IS_EMPTY_1(...) 0
#define IS_EMPTY_NON_FUNCTION_C(...) ()

#define IS_BEGIN_PARENS(...) P_FIRST(P_CAT(P_IS_VARIADIC_R_, P_IS_VARIADIC_C __VA_ARGS__))

#define P_IS_VARIADIC_R_1 1,
#define P_IS_VARIADIC_R_P_IS_VARIADIC_C 0,
#define P_IS_VARIADIC_C(...) 1

#define P_FIRST(...) P_FIRST_(__VA_ARGS__, )
#define P_FIRST_(a, ...) a

#define P_CAT(a, ...) P_CAT_(a, __VA_ARGS__)
#define P_CAT_(a, ...) a##__VA_ARGS__

#define P_COMPL(b) P_CAT(P_COMPL_, b)
#define P_COMPL_0 1
#define P_COMPL_1 0

#define ACTIVATION_INIT(elem) ACTIVATION_INIT_##elem()
#define ACTIVATION_APPLY(elem, i) ACTIVATION_APPLY_##elem(i)

#define PROCESS_ACTIVATION(elem, i) \
    ACTIVATION_INIT(elem);          \
    ACTIVATION_APPLY(elem, i)

#define PROCESS_ACTIVATIONS(op, i) PROCESS_ACTIVATIONS_(op)(i)
#define PROCESS_ACTIVATIONS_(op) PROCESS_##op##_ACTIVATIONS
#define HAS_ACTIVATIONS(op) P_COMPL(IS_EMPTY(PROCESS_ACTIVATIONS(op, 0)))

#define BCAST_OP P_CAT(BCAST_OP_, BCAST_INPUT)
#define OTHER_OP P_CAT(BCAST_OP_, P_COMPL(BCAST_INPUT))
#define BCAST_OP_0 LHS
#define BCAST_OP_1 RHS

#define PREPROCESS(op, ...) P_CAT(PREPROCESS_, HAS_ACTIVATIONS(op))(op, __VA_ARGS__)
#define PREPROCESS_0(...)
#define PREPROCESS_1(op, cb_pre, cb_post, cb_out, per_core_block_size) \
    do {                                                               \
        using namespace ckernel;                                       \
                                                                       \
        reconfig_data_format_srca(/*old*/ cb_post, /*new*/ cb_pre);    \
        pack_reconfig_data_format(/*old*/ cb_out, /*new*/ cb_post);    \
                                                                       \
        cb_wait_front(cb_pre, per_core_block_size);                    \
        cb_reserve_back(cb_post, per_core_block_size);                 \
                                                                       \
        tile_regs_acquire();                                           \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {           \
            copy_tile_to_dst_init_short();                             \
            copy_tile(cb_pre, i, i);                                   \
            PROCESS_ACTIVATIONS(op, i);                                \
        }                                                              \
        tile_regs_commit();                                            \
                                                                       \
        tile_regs_wait();                                              \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {           \
            pack_tile(i, cb_post);                                     \
        }                                                              \
        tile_regs_release();                                           \
                                                                       \
        cb_pop_front(cb_pre, per_core_block_size);                     \
        cb_push_back(cb_post, per_core_block_size);                    \
                                                                       \
        reconfig_data_format_srca(/*old*/ cb_pre, /*new*/ cb_post);    \
        pack_reconfig_data_format(/*old*/ cb_post, /*new*/ cb_out);    \
    } while (0)
