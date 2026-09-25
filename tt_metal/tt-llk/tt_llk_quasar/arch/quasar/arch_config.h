// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ckernel_instr_params.h>
#include <ckernel_proj_params.h>

#include <cstdint>

#ifndef ENABLE_TENSIX_GATHER
#define ENABLE_TENSIX_GATHER 0
#endif

namespace ckernel::arch
{
constexpr std::uint32_t fpu_rows       = MATH_ROWS;
constexpr std::uint32_t dest_row_group = 8;

constexpr std::uint32_t mov_rows_encoding(const std::uint32_t rows)
{
    return rows == 1 ? 0x0 : rows == 4 ? 0x1 : rows == 8 ? 0x2 : 0x3;
}

static_assert(mov_rows_encoding(1) == p_mov_src_to_dest::MOV_1_ROW);
static_assert(mov_rows_encoding(4) == p_mov_src_to_dest::MOV_4_ROWS);
static_assert(mov_rows_encoding(8) == p_mov_src_to_dest::MOV_8_ROWS);
static_assert(mov_rows_encoding(4) == p_movd2a::MOV_4_ROWS && mov_rows_encoding(8) == p_movd2a::MOV_8_ROWS);
static_assert(mov_rows_encoding(4) == p_movd2b::MOV_4_ROWS && mov_rows_encoding(8) == p_movd2b::MOV_8_ROWS);
static_assert(mov_rows_encoding(4) == p_movb2a::MOV_4_ROWS && mov_rows_encoding(8) == p_movb2a::MOV_8_ROWS);

constexpr std::uint32_t mov_fpu_rows = mov_rows_encoding(fpu_rows);
constexpr bool has_fp4_pack          = (ENABLE_FP4_PACKING != 0);
constexpr bool has_int8_pack         = (ENABLE_INT8_PACKING != 0);
constexpr bool has_int32_1           = (INT_32_1_ENABLED != 0);
constexpr bool has_tensix_gather     = (ENABLE_TENSIX_GATHER != 0);

static_assert(fpu_rows == 4 || fpu_rows == 8, "unsupported MATH_ROWS: the math LLKs assume an FPU width of 4 or 8");
static_assert(dest_row_group % fpu_rows == 0, "an FPU issue must divide the dest row group");
} // namespace ckernel::arch
