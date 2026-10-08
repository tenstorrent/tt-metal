// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

// Physical LHS means the tensor in c_0, not necessarily the mathematical LHS.
// This is a FORMAT reference, not necessarily the buffer supplying the next tile:
// binary_ng_program_factory gives the LHS broadcast temporary (c_5) the same
// format as the original LHS (c_0). With LHS activation, use its intermediate
// (c_3), whose format can differ from c_0 (e.g. LOGADDEXP).
#define BINARY_PHYSICAL_LHS_FORMAT_CB (HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : tt::CBIndex::c_0)

// FPU scalar-first kernels start SrcA from physical RHS (the scalar), and keep
// that operand order in binary_tiles_init. Host activation defines are already
// mapped to physical slots, so RHS selects c_1/c_4 here, not the logical RHS.
// SFPU scalar-first kernels instead load c_0 first and swap DST operand indices;
// their preprocessing must continue to restore BINARY_PHYSICAL_LHS_FORMAT_CB.
#if SCALAR_IS_LHS
#define BINARY_FPU_SRCA_FORMAT_CB (HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : tt::CBIndex::c_1)
#else
#define BINARY_FPU_SRCA_FORMAT_CB BINARY_PHYSICAL_LHS_FORMAT_CB
#endif

#if defined(TRISC_UNPACK) && !HAS_ACTIVATIONS(LHS) && HAS_ACTIVATIONS(RHS)
// With only RHS activation, preprocessing restores SrcA using c_0's settings,
// even when LHS broadcast tiles are in c_5. If c_5 exists, its formats and tile
// geometry must match c_0 so that restoration is safe. The 0xff value means
// c_5 is unused, so there is nothing to check.
static_assert(
    unpack_src_format[5] == 0xff ||
        (unpack_src_format[0] == unpack_src_format[5] && unpack_dst_format[0] == unpack_dst_format[5] &&
         unpack_tile_num_faces[0] == unpack_tile_num_faces[5] &&
         unpack_tile_face_r_dim[0] == unpack_tile_face_r_dim[5] && unpack_partial_face[0] == unpack_partial_face[5] &&
         unpack_narrow_tile[0] == unpack_narrow_tile[5] && unpack_tile_r_dim[0] == unpack_tile_r_dim[5] &&
         unpack_tile_c_dim[0] == unpack_tile_c_dim[5] && unpack_tile_size[0] == unpack_tile_size[5]),
    "binary_ng: LHS broadcast buffer no longer matches the shared SrcA format reference");
#endif

#define BCAST_OP P_CAT(BCAST_OP_, BCAST_INPUT)
#define OTHER_OP P_CAT(BCAST_OP_, P_COMPL(BCAST_INPUT))
#define BCAST_OP_0 LHS
#define BCAST_OP_1 RHS

// In that build, the SFPU tile API takes two extra runtime-arg scalars
// (rtol/atol IEEE-754 bits) which are read once at the top of kernel_main and
// then forwarded into the inlined process_tile helpers via these macros.
#ifdef ISCLOSE_OP
#define ISCLOSE_RT_ARG_PARAMS , uint32_t rtol_bits, uint32_t atol_bits
#define ISCLOSE_RT_ARG_FWD , rtol_bits, atol_bits
#else
#define ISCLOSE_RT_ARG_PARAMS
#define ISCLOSE_RT_ARG_FWD
#endif
