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

#define ACTIVATION_INIT_TANH tanh_tile_init
#define ACTIVATION_APPLY_TANH tanh_tile

#define ACTIVATION_INIT_LOG2 log_with_base_tile_init
#define ACTIVATION_APPLY_LOG2(i) log_with_base_tile(i, 0x3dc5u)

#define ACTIVATION_INIT_LOG10 log_with_base_tile_init
#define ACTIVATION_APPLY_LOG10(i) log_with_base_tile(i, 0x36f3u)

#define ACTIVATION_INIT_EXP exp_tile_init<false>
#define ACTIVATION_APPLY_EXP exp_tile<false>

#define ACTIVATION_INIT_EXP2 exp2_tile_init
#define ACTIVATION_APPLY_EXP2 exp2_tile

#define ACTIVATION_INIT_EXPM1 expm1_tile_init
#define ACTIVATION_APPLY_EXPM1 expm1_tile

#define ACTIVATION_INIT_RECIP recip_tile_init
#define ACTIVATION_APPLY_RECIP recip_tile

#define ACTIVATION_INIT_GELU gelu_tile_init<false>
#define ACTIVATION_APPLY_GELU gelu_tile<false>

#define ACTIVATION_INIT_SQRT sqrt_tile_init
#define ACTIVATION_APPLY_SQRT sqrt_tile

#define ACTIVATION_INIT_SIGMOID sigmoid_tile_init
#define ACTIVATION_APPLY_SIGMOID sigmoid_tile

#define ACTIVATION_INIT_SIN sin_tile_init
#define ACTIVATION_APPLY_SIN sin_tile

#define ACTIVATION_INIT_COS cos_tile_init
#define ACTIVATION_APPLY_COS cos_tile

#define ACTIVATION_INIT_TAN tan_tile_init
#define ACTIVATION_APPLY_TAN tan_tile

#define ACTIVATION_INIT_ASIN asin_tile_init
#define ACTIVATION_APPLY_ASIN asin_tile

#define ACTIVATION_INIT_ACOS acos_tile_init
#define ACTIVATION_APPLY_ACOS acos_tile

#define ACTIVATION_INIT_ATAN atan_tile_init
#define ACTIVATION_APPLY_ATAN atan_tile

#define ACTIVATION_INIT_ABS abs_tile_init
#define ACTIVATION_APPLY_ABS abs_tile

#define ACTIVATION_INIT_SIGN sign_tile_init
#define ACTIVATION_APPLY_SIGN sign_tile

#define ACTIVATION_INIT_SIGNBIT signbit_tile_init
#define ACTIVATION_APPLY_SIGNBIT signbit_tile

#define ACTIVATION_INIT_RSQRT rsqrt_tile_init<false>
#define ACTIVATION_APPLY_RSQRT rsqrt_tile<false>

#define ACTIVATION_INIT_RELU6 relu_max_tile_init
#define ACTIVATION_APPLY_RELU6(i) relu_max_tile(i, 0x40c00000u)

#define ACTIVATION_INIT_ERF erf_tile_init
#define ACTIVATION_APPLY_ERF erf_tile

#define ACTIVATION_INIT_ERFC erfc_tile_init
#define ACTIVATION_APPLY_ERFC erfc_tile

#define ACTIVATION_INIT_ISINF isinf_tile_init
#define ACTIVATION_APPLY_ISINF isinf_tile

#define ACTIVATION_INIT_ISPOSINF isposinf_tile_init
#define ACTIVATION_APPLY_ISPOSINF isposinf_tile

#define ACTIVATION_INIT_ISNEGINF isneginf_tile_init
#define ACTIVATION_APPLY_ISNEGINF isneginf_tile

#define ACTIVATION_INIT_ISNAN isnan_tile_init
#define ACTIVATION_APPLY_ISNAN isnan_tile

#define ACTIVATION_INIT_ISFINITE isfinite_tile_init
#define ACTIVATION_APPLY_ISFINITE isfinite_tile

#define ACTIVATION_INIT_LOGICAL_NOT_UNARY logical_not_unary_tile_init
#define ACTIVATION_APPLY_LOGICAL_NOT_UNARY logical_not_unary_tile

#define ACTIVATION_INIT_ERFINV erfinv_tile_init
#define ACTIVATION_APPLY_ERFINV erfinv_tile

#define ACTIVATION_INIT_I0 i0_tile_init
#define ACTIVATION_APPLY_I0 i0_tile

#define ACTIVATION_INIT_I1 i1_tile_init
#define ACTIVATION_APPLY_I1 i1_tile

#define ACTIVATION_INIT_SILU silu_tile_init
#define ACTIVATION_APPLY_SILU silu_tile

#define ACTIVATION_INIT_NEG negative_tile_init
#define ACTIVATION_APPLY_NEG negative_tile

#define ACTIVATION_INIT_BITWISE_NOT bitwise_not_tile_init
#define ACTIVATION_APPLY_BITWISE_NOT bitwise_not_tile

#define ACTIVATION_INIT_FLOOR floor_tile_init
#define ACTIVATION_APPLY_FLOOR floor_tile

#define ACTIVATION_INIT_CEIL ceil_tile_init
#define ACTIVATION_APPLY_CEIL ceil_tile

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
            copy_tile_to_dst_init_short(cb_pre);                       \
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
