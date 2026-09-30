// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "tensix_types.h"

namespace ckernel
{
namespace sfpu
{

// ============================================================================
// SFPU forward-substitution triangle solve of one 32x32 tile
// ============================================================================
//
//   L X = RHS with L unit lower-triangular:  X[r] = RHS[r] - sum_{c < r} L[r][c] * X[c]
//
// L stays in L1: the RISC reads each strict-lower entry and splats it across the 32 lanes with SFPLOADI, so L is
// never staged in DEST or rounded. RHS is DEST tile dst_in; X is left in DEST tile dst_out in standard tile layout.
// The unit diagonal is implicit and never read. Blackhole only: the SFPMAD accumulation chain relies on the
// Blackhole scoreboard.
//
// DEST addressing (32-bit accumulation): a 32x32 tile is 64 DEST rows of 16 datums, faces f0..f3 at rows 0, 16, 32,
// 48. One SFPLOAD/SFPSTORE moves a 4-row x 8-column block; the four blocks of tile rows 4g..4g+3 sit at
// base + {0, 2, 16, 18} (face 0 / face 1, even / odd column half) with base = (g & 3) * 4 for faces 0/1 and
// 32 + (g & 3) * 4 for faces 2/3. Rows are solved in groups of four held in LREG0..3: every already-solved column
// contributes a rank-1 update to the live group, then the group's own triangle is applied. Solved rows are stashed
// row-oriented in dst_out, one block slot each, and transposed back in place at the end.

constexpr std::uint32_t TRIANGLE_SOLVE_TILE_DIM        = TILE_R_DIM;
constexpr std::uint32_t TRIANGLE_SOLVE_FACE_DIM        = FACE_R_DIM;
constexpr std::uint32_t TRIANGLE_SOLVE_ROWS_PER_GROUP  = 4;
constexpr std::uint32_t TRIANGLE_SOLVE_GROUPS          = TRIANGLE_SOLVE_TILE_DIM / TRIANGLE_SOLVE_ROWS_PER_GROUP;
constexpr std::uint32_t TRIANGLE_SOLVE_FACE_ROWS       = FACE_R_DIM; // DEST rows per face: 16 datums per row
constexpr std::uint32_t TRIANGLE_SOLVE_FACE_PAIR_ROWS  = 2 * TRIANGLE_SOLVE_FACE_ROWS;
constexpr std::uint32_t TRIANGLE_SOLVE_ODD_COLUMN_HALF = 2; // SFPLOAD address step to columns 8..15 of a face
// The four block slots of a row group: face 0 even half, face 0 odd half, face 1 even half, face 1 odd half.
constexpr std::uint32_t TRIANGLE_SOLVE_BLOCK_OFF[TRIANGLE_SOLVE_ROWS_PER_GROUP] = {
    0, TRIANGLE_SOLVE_ODD_COLUMN_HALF, TRIANGLE_SOLVE_FACE_ROWS, TRIANGLE_SOLVE_FACE_ROWS + TRIANGLE_SOLVE_ODD_COLUMN_HALF};
// DEST rows one 32x32 tile occupies, from the same shift set_dst_write_addr uses.
constexpr std::uint32_t TRIANGLE_SOLVE_DEST_TILE_ROWS = 1u << DstTileSizeLog2[DstTileShape::Tile32x32];
static_assert(TRIANGLE_SOLVE_DEST_TILE_ROWS == 2 * TRIANGLE_SOLVE_FACE_PAIR_ROWS, "the solve's DEST layout assumes four 16-row faces per 32x32 tile");

constexpr std::uint32_t TRIANGLE_SOLVE_SIGN_BIT_FP32 = 0x80000000u;
constexpr std::uint32_t TRIANGLE_SOLVE_SIGN_BIT_BF16 = 0x8000u;

template <DataFormat L_FORMAT>
using _triangle_solve_l_elem_t_ = std::conditional_t<L_FORMAT == DataFormat::Float32, std::uint32_t, std::uint16_t>;

/**
 * @brief Element index of (row, col) in a 32x32 tile in standard TILE layout: four row-major 16x16 faces [f0, f1, f2, f3].
 *
 * @param row: Tile row, 0..31.
 * @param col: Tile column, 0..31.
 */
inline constexpr std::uint32_t _triangle_solve_elem_(const std::uint32_t row, const std::uint32_t col)
{
    const std::uint32_t face = (row / TRIANGLE_SOLVE_FACE_DIM) * 2 + col / TRIANGLE_SOLVE_FACE_DIM;
    return face * TRIANGLE_SOLVE_FACE_DIM * TRIANGLE_SOLVE_FACE_DIM + (row % TRIANGLE_SOLVE_FACE_DIM) * TRIANGLE_SOLVE_FACE_DIM + col % TRIANGLE_SOLVE_FACE_DIM;
}

/**
 * @brief DEST offset (within a tile) of the first block of a row group.
 *
 * @param group: Row group index, 0..7 (tile rows 4 * group .. 4 * group + 3).
 */
inline constexpr std::uint32_t _triangle_solve_group_base_(const std::uint32_t group)
{
    return (group % 4) * TRIANGLE_SOLVE_ROWS_PER_GROUP + (group / 4) * TRIANGLE_SOLVE_FACE_PAIR_ROWS;
}

/**
 * @brief DEST offset (within a tile) of the block slot where a logical row is stashed row-oriented during the solve.
 *
 * @param row: Tile row, 0..31.
 */
inline constexpr std::uint32_t _triangle_solve_row_off_(const std::uint32_t row)
{
    return _triangle_solve_group_base_(row / TRIANGLE_SOLVE_ROWS_PER_GROUP) + TRIANGLE_SOLVE_BLOCK_OFF[row % TRIANGLE_SOLVE_ROWS_PER_GROUP];
}

/**
 * @brief Splat -L (or L when L_NEGATED, i.e. the entry is already -L) into LREG7 across all lanes from the raw bits of one L element.
 *
 * fp32 takes two immediates (upper and lower halves); bf16 is the upper half of an fp32, so one FLOATB immediate restores the exact
 * value. The subtraction of the update is folded in here by flipping the sign bit, which keeps the SFPMAD a plain accumulate.
 *
 * @tparam L_FORMAT: Format of the L tile, values = <Float32/Float16_b>
 * @tparam L_NEGATED: L's strict-lower entries are supplied negated
 * @param bits: Raw bits of the element as read from L1.
 */
template <DataFormat L_FORMAT, bool L_NEGATED>
inline void _triangle_solve_load_l_(std::uint32_t bits)
{
    static_assert(L_FORMAT == DataFormat::Float32 || L_FORMAT == DataFormat::Float16_b, "the triangle solve reads L as Float32 or Float16_b");
    if constexpr (L_FORMAT == DataFormat::Float32)
    {
        if constexpr (!L_NEGATED)
        {
            bits ^= TRIANGLE_SOLVE_SIGN_BIT_FP32;
        }
        TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
        TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xFFFF);
    }
    else
    {
        if constexpr (!L_NEGATED)
        {
            bits ^= TRIANGLE_SOLVE_SIGN_BIT_BF16;
        }
        TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, bits);
    }
}

/**
 * @brief Rank-1 update of the live row group (LREG0..3) from one solved column: LREG r -= L[row0 + r][col] * X[col].
 *
 * X[col] is loaded once into LREG4; the four L entries sit one face row (16 elements) apart in L1 and their reads issue in the
 * shadow of the SFPU instructions.
 *
 * @tparam L_FORMAT: Format of the L tile, values = <Float32/Float16_b>
 * @tparam L_NEGATED: L's strict-lower entries are supplied negated
 * @param l_col: L1 pointer to L[row0][col].
 * @param x_addr: DEST address of the stashed row X[col].
 */
template <DataFormat L_FORMAT, bool L_NEGATED>
inline void _triangle_solve_apply_prev_col_(volatile tt_l1_ptr _triangle_solve_l_elem_t_<L_FORMAT>* l_col, const std::uint32_t x_addr)
{
    const std::uint32_t b0 = l_col[0 * TRIANGLE_SOLVE_FACE_DIM];
    TT_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, x_addr);
    const std::uint32_t b1 = l_col[1 * TRIANGLE_SOLVE_FACE_DIM];
    // SFPLOAD and SFPLOADI are both load-class instructions and must not issue in adjacent slots; the SFPLOADIs that
    // follow then also cover the SFPLOAD -> SFPMAD load-use of LREG4.
    TTI_SFPNOP;
    _triangle_solve_load_l_<L_FORMAT, L_NEGATED>(b0);
    if constexpr (L_FORMAT == DataFormat::Float16_b)
    {
        // bf16 splats with one immediate instead of two: pad so the SFPLOAD -> SFPMAD distance stays that of the fp32 stream.
        TTI_SFPNOP;
    }
    const std::uint32_t b2 = l_col[2 * TRIANGLE_SOLVE_FACE_DIM];
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    _triangle_solve_load_l_<L_FORMAT, L_NEGATED>(b1);
    const std::uint32_t b3 = l_col[3 * TRIANGLE_SOLVE_FACE_DIM];
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    _triangle_solve_load_l_<L_FORMAT, L_NEGATED>(b2);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    _triangle_solve_load_l_<L_FORMAT, L_NEGATED>(b3);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

/**
 * @brief Load the four rows of a row group into LREG0..3, row-oriented.
 *
 * The transpose before the block loads and the one after them turn the four 4x8 blocks into four 32-lane rows.
 *
 * @param base: DEST offset of the row group's first block.
 */
inline void _triangle_solve_load_group_(const std::uint32_t base)
{
    TTI_SFPTRANSP(0, 0, 0, 0);
    TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[0]);
    TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[1]);
    TT_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[2]);
    TT_SFPLOAD(p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[3]);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

/**
 * @brief Transpose a row group from its row-oriented stash back to standard tile layout, in place.
 *
 * The four blocks are read before any of them is written, and row groups are disjoint, so nothing is clobbered before it is read.
 *
 * @param base: DEST offset of the row group's first block.
 */
inline void _triangle_solve_restore_group_layout_(const std::uint32_t base)
{
    TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[0]);
    TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[1]);
    TT_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[2]);
    TT_SFPLOAD(p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[3]);
    TTI_SFPTRANSP(0, 0, 0, 0);
    TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[0]);
    TT_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[1]);
    TT_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[2]);
    TT_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[3]);
}

/**
 * @brief Solve L X = RHS for one 32x32 tile by forward substitution, L unit lower-triangular and read in place from L1.
 *
 * @tparam L_FORMAT: Format of the L tile in L1, values = <Float32/Float16_b>
 * @tparam L_NEGATED: L's strict-lower entries are supplied negated (the tile holds -L below the diagonal)
 * @param dst_in: DEST tile index holding RHS.
 * @param dst_out: DEST tile index that receives X in standard tile layout; must differ from dst_in.
 * @param l1_base: L1 byte address of the L tile in standard TILE layout; must stay resident for the whole call.
 * @note DEST is addressed absolutely (tile index * TRIANGLE_SOLVE_DEST_TILE_ROWS): bracket the call with
 *       @ref _llk_math_eltwise_sfpu_start_ at DEST base 0 and @ref _llk_math_eltwise_sfpu_done_, with 32-bit destination
 *       accumulation enabled and ADDR_MOD_7 programmed by @ref _llk_math_eltwise_binary_sfpu_init_. Writes LREG0..4 and
 *       LREG7.
 */
template <DataFormat L_FORMAT, bool L_NEGATED>
inline void _triangle_solve_tile_(const std::uint32_t dst_in, const std::uint32_t dst_out, const std::uint32_t l1_base)
{
    using l_elem_t                          = _triangle_solve_l_elem_t_<L_FORMAT>;
    volatile tt_l1_ptr l_elem_t* const tile = reinterpret_cast<volatile tt_l1_ptr l_elem_t*>(l1_base);
    const std::uint32_t in_base             = dst_in * TRIANGLE_SOLVE_DEST_TILE_ROWS;
    const std::uint32_t out_base            = dst_out * TRIANGLE_SOLVE_DEST_TILE_ROWS;

    for (std::uint32_t group = 0; group < TRIANGLE_SOLVE_GROUPS; group++)
    {
        _triangle_solve_load_group_(in_base + _triangle_solve_group_base_(group));

        // Columns of the previous groups. Columns 0..15 and 16..row0-1 live in different faces, so the element pointer
        // restarts at column 16 instead of walking across the face boundary.
        const std::uint32_t row0 = group * TRIANGLE_SOLVE_ROWS_PER_GROUP;
        if (row0 > 0)
        {
            const std::uint32_t face0_cols             = row0 < TRIANGLE_SOLVE_FACE_DIM ? row0 : TRIANGLE_SOLVE_FACE_DIM;
            volatile tt_l1_ptr l_elem_t* const l_face0 = tile + _triangle_solve_elem_(row0, 0);
            for (std::uint32_t col = 0; col < face0_cols; col++)
            {
                _triangle_solve_apply_prev_col_<L_FORMAT, L_NEGATED>(l_face0 + col, out_base + _triangle_solve_row_off_(col));
            }
            if (row0 > TRIANGLE_SOLVE_FACE_DIM)
            {
                volatile tt_l1_ptr l_elem_t* const l_face1 = tile + _triangle_solve_elem_(row0, TRIANGLE_SOLVE_FACE_DIM);
                for (std::uint32_t col = TRIANGLE_SOLVE_FACE_DIM; col < row0; col++)
                {
                    _triangle_solve_apply_prev_col_<L_FORMAT, L_NEGATED>(l_face1 + (col - TRIANGLE_SOLVE_FACE_DIM), out_base + _triangle_solve_row_off_(col));
                }
            }
        }

        // The group's own triangle: row r uses X[row0 + k], k < r, still live in LREG k; each row is stashed once solved.
        TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, out_base + _triangle_solve_row_off_(row0 + 0));
        {
            volatile tt_l1_ptr l_elem_t* const l_row = tile + _triangle_solve_elem_(row0 + 1, row0);
            _triangle_solve_load_l_<L_FORMAT, L_NEGATED>(l_row[0]);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPMAD_MOD1_OFFSET_NONE);
            TT_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, out_base + _triangle_solve_row_off_(row0 + 1));
        }
        {
            volatile tt_l1_ptr l_elem_t* const l_row = tile + _triangle_solve_elem_(row0 + 2, row0);
            const std::uint32_t b0                   = l_row[0];
            _triangle_solve_load_l_<L_FORMAT, L_NEGATED>(b0);
            const std::uint32_t b1 = l_row[1];
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPMAD_MOD1_OFFSET_NONE);
            _triangle_solve_load_l_<L_FORMAT, L_NEGATED>(b1);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPMAD_MOD1_OFFSET_NONE);
            TT_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, out_base + _triangle_solve_row_off_(row0 + 2));
        }
        {
            volatile tt_l1_ptr l_elem_t* const l_row = tile + _triangle_solve_elem_(row0 + 3, row0);
            const std::uint32_t b0                   = l_row[0];
            _triangle_solve_load_l_<L_FORMAT, L_NEGATED>(b0);
            const std::uint32_t b1 = l_row[1];
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPMAD_MOD1_OFFSET_NONE);
            _triangle_solve_load_l_<L_FORMAT, L_NEGATED>(b1);
            const std::uint32_t b2 = l_row[2];
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPMAD_MOD1_OFFSET_NONE);
            _triangle_solve_load_l_<L_FORMAT, L_NEGATED>(b2);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPMAD_MOD1_OFFSET_NONE);
            TT_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, out_base + _triangle_solve_row_off_(row0 + 3));
        }
    }

    for (std::uint32_t group = 0; group < TRIANGLE_SOLVE_GROUPS; group++)
    {
        _triangle_solve_restore_group_layout_(out_base + _triangle_solve_group_base_(group));
    }
}

/**
 * @brief Init for @ref _triangle_solve_tile_. The solve needs no state beyond the ADDR_MOD_7 of @ref _llk_math_eltwise_binary_sfpu_init_.
 */
inline void _triangle_solve_init_()
{
}

} // namespace sfpu
} // namespace ckernel
