// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
// Angle brackets so a variant directory (arch/<variant>/) placed first on the include path can shadow it.
#include <ckernel_proj_params.h>

// The only reader of the generated project macros. Kernels use these constants instead.
namespace ckernel::arch
{
constexpr std::uint32_t fpu_rows       = MATH_ROWS; // rows one FPU instruction covers
constexpr std::uint32_t dest_row_group = 8;         // dest/src row layout, fixed on every variant

static_assert(fpu_rows == 4 || fpu_rows == 8, "the math LLKs support a 4-row or 8-row FPU");
static_assert(dest_row_group % fpu_rows == 0, "FPU rows must divide the dest row group");

constexpr std::uint32_t mov_rows_encoding(const std::uint32_t rows)
{
    return rows == 1 ? p_mov_src_to_dest::MOV_1_ROW : rows == 4 ? p_mov_src_to_dest::MOV_4_ROWS : p_mov_src_to_dest::MOV_8_ROWS;
}

static_assert(mov_rows_encoding(4) == p_movd2a::MOV_4_ROWS && mov_rows_encoding(8) == p_movd2a::MOV_8_ROWS);
static_assert(mov_rows_encoding(4) == p_movd2b::MOV_4_ROWS && mov_rows_encoding(8) == p_movd2b::MOV_8_ROWS);
static_assert(mov_rows_encoding(4) == p_movb2a::MOV_4_ROWS && mov_rows_encoding(8) == p_movb2a::MOV_8_ROWS);

constexpr std::uint32_t mov_fpu_rows = mov_rows_encoding(fpu_rows);

constexpr bool has_fp4_pack  = (ENABLE_FP4_PACKING != 0);
constexpr bool has_int8_pack = (ENABLE_INT8_PACKING != 0);
constexpr bool has_int32_1   = (INT_32_1_ENABLED != 0);
} // namespace ckernel::arch
