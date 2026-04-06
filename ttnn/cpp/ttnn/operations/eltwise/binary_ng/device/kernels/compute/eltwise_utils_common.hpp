// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

#define PROCESS_ACTIVATIONS(op, i) PROCESS_ACTIVATIONS_(op)(i)
#define PROCESS_ACTIVATIONS_(op) PROCESS_##op##_ACTIVATIONS
#define HAS_ACTIVATIONS(op) P_COMPL(IS_EMPTY(PROCESS_ACTIVATIONS(op, 0)))

#define BCAST_OP P_CAT(BCAST_OP_, BCAST_INPUT)
#define OTHER_OP P_CAT(BCAST_OP_, P_COMPL(BCAST_INPUT))
#define BCAST_OP_0 LHS
#define BCAST_OP_1 RHS
