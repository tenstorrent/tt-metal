// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Single translation layer from the generated per-project macros in
 * ckernel_proj_params.h to typed C++ constants.
 *
 * This is the only header permitted to include ckernel_proj_params.h or to name
 * a raw project macro; infra/check_arch_config.py enforces both. Everything that
 * varies between Quasar parts is a named constant here, so a new part costs a
 * regenerated ckernel_proj_params.h plus at most one capability constant — never
 * a second copy of a kernel.
 *
 * Naming rule: name a constant for the capability, never for the part.
 * `has_mxfp4_2x_replay`, not `is_quasar_n4`. A capability constant is reused by
 * the next part that shares the trait, so that part adds no branches at all; a
 * part-named constant is never reusable and multiplies conditionals instead.
 */
#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
// Angle brackets, deliberately: this header is supplied by the build, not by a neighbouring file.
// Which configuration under tt_llk_quasar/proj/ it resolves to is chosen by the include path. A
// quoted include would search this file's own directory first, so a stray copy landing in
// common/inc would silently win over the selected configuration.
#include <ckernel_proj_params.h>

namespace ckernel::arch
{

// ---------------------------------------------------------------------------
// Geometry
//
// MAX_FPU_ROWS (common/tensor_shape.h) used to carry both of the constants
// below under one name and one literal 8. They are independent: one follows the
// FPU, the other follows the dest/src register layout. Keep them apart.
// ---------------------------------------------------------------------------

// Rows covered by one FPU issue (MVMUL) or one MOV*. Eight on the base part,
// four on narrow-FPU parts. The only geometry value that varies per part.
//
// Which definition of MATH_ROWS arrives here is a property of the build configuration:
// see tt_llk_quasar/proj/. This site does not know or care which one it got.
constexpr std::uint32_t fpu_rows = MATH_ROWS;

// Granularity of the dest/src row layout: sub-eight-row faces sit sparsely, one
// every eight rows, whatever the FPU width. Fixed on every part.
constexpr std::uint32_t dest_row_group = 8;

// True when one dest row group takes more than one FPU issue, i.e. the FPU is narrower than the
// layout granularity. Tiny-tile matmul then needs its own replay image: the full-tile image walks a
// 16-row face, and no window of it walks an eight-row one.
constexpr bool fpu_splits_dest_row_group = dest_row_group > fpu_rows;

// ---------------------------------------------------------------------------
// Instruction encodings derived from the geometry
// ---------------------------------------------------------------------------

// Row-count field shared by MOVD2A / MOVD2B / MOVB2A / MOVB2D / MOVA2D. The
// four parameter structs encode it identically; the static_asserts below hold
// them to that.
constexpr std::uint32_t mov_rows_encoding(const std::uint32_t rows)
{
    return rows == 1 ? 0x0 : rows == 4 ? 0x1 : rows == 8 ? 0x2 : 0x3 /* 16 */;
}

static_assert(mov_rows_encoding(1) == p_mov_src_to_dest::MOV_1_ROW);
static_assert(mov_rows_encoding(4) == p_mov_src_to_dest::MOV_4_ROWS);
static_assert(mov_rows_encoding(8) == p_mov_src_to_dest::MOV_8_ROWS);
static_assert(mov_rows_encoding(4) == p_movd2a::MOV_4_ROWS && mov_rows_encoding(8) == p_movd2a::MOV_8_ROWS);
static_assert(mov_rows_encoding(4) == p_movd2b::MOV_4_ROWS && mov_rows_encoding(8) == p_movd2b::MOV_8_ROWS);
static_assert(mov_rows_encoding(4) == p_movb2a::MOV_4_ROWS && mov_rows_encoding(8) == p_movb2a::MOV_8_ROWS);

// The MOV row-count field matching this part's FPU width. Use this instead of
// spelling MOV_8_ROWS or MOV_4_ROWS at a site whose width follows the FPU.
constexpr std::uint32_t mov_fpu_rows = mov_rows_encoding(fpu_rows);

// ---------------------------------------------------------------------------
// Capabilities
//
// One constant per hardware capability, named for the capability. Add a new IP
// by setting these, not by forking a kernel.
// ---------------------------------------------------------------------------

// Records the MXFP4_2x matmul as a seven-MVMUL replay image. Parts without it
// reach 2x through direct indexing instead (see _llk_math_matmul_init_ in
// llk_lib/llk_math_matmul.h).
constexpr bool has_mxfp4_2x_replay = (fpu_rows == 8);

// Packer emits FP4; Int8 packing; 32-bit integer datapath variant. Already
// generated per project, consumed here so a future user does not reach for the
// raw macro.
constexpr bool has_fp4_pack  = (ENABLE_FP4_PACKING != 0);
constexpr bool has_int8_pack = (ENABLE_INT8_PACKING != 0);
constexpr bool has_int32_1   = (INT_32_1_ENABLED != 0);

// ---------------------------------------------------------------------------
// Invariants every part must satisfy
// ---------------------------------------------------------------------------

static_assert(fpu_rows == 4 || fpu_rows == 8, "unsupported MATH_ROWS: the math LLKs assume an FPU width of 4 or 8");
static_assert(dest_row_group % fpu_rows == 0, "an FPU issue must divide the dest row group");

} // namespace ckernel::arch
