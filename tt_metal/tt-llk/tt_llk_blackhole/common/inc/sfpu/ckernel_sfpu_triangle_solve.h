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
// The unit diagonal is implicit and never read. Blackhole only: the stream carries no SFPNOPs. SFPLOAD and
// SFPLOADI results are ready the next cycle, so a splat may follow the X load and the first SFPMAD may follow the
// last splat directly (the adjacency the sfpi compiler itself emits), and the SFPMAD -> SFPMAD/SFPSTORE/SFPTRANSP
// dependences of the 2-cycle SFPMAD result are stalled by the Blackhole scoreboard (none of those consumers is in
// the Blackhole errata set that needs an explicit NOP).
//
// DEST addressing (32-bit accumulation): a 32x32 tile is 64 DEST rows of 16 datums, faces f0..f3 at rows 0, 16, 32,
// 48. One SFPLOAD/SFPSTORE moves 32 lanes, four rows by eight columns of one face: the even columns at an even
// address, the odd columns (1, 3, ..., 15) of the same rows at address + 2 (lane mapping
// col = (lane & 7) * 2 + ((addr & 2) ? 1 : 0)). The four blocks of tile rows 4g..4g+3 sit at base + {0, 2, 16, 18}
// (face 0 / face 1, even / odd columns) with base = (g & 3) * 4 for faces 0/1 and 32 + (g & 3) * 4 for faces 2/3. Rows are solved in groups of four held in
// LREG0..3: every already-solved column contributes a rank-1 update to the live group, then the group's own triangle is applied. Solved rows are stashed
// row-oriented in dst_out, one block slot each, and transposed back in place at the end.

constexpr std::uint32_t TRIANGLE_SOLVE_TILE_DIM        = TILE_R_DIM;
constexpr std::uint32_t TRIANGLE_SOLVE_FACE_DIM        = FACE_R_DIM;
constexpr std::uint32_t TRIANGLE_SOLVE_ROWS_PER_GROUP  = 4;
constexpr std::uint32_t TRIANGLE_SOLVE_GROUPS          = TRIANGLE_SOLVE_TILE_DIM / TRIANGLE_SOLVE_ROWS_PER_GROUP;
constexpr std::uint32_t TRIANGLE_SOLVE_GROUPS_PER_FACE = TRIANGLE_SOLVE_FACE_DIM / TRIANGLE_SOLVE_ROWS_PER_GROUP;
constexpr std::uint32_t TRIANGLE_SOLVE_FACES_PER_ROW   = TRIANGLE_SOLVE_TILE_DIM / TRIANGLE_SOLVE_FACE_DIM;
constexpr std::uint32_t TRIANGLE_SOLVE_FACE_ROWS       = FACE_R_DIM; // DEST rows per face: 16 datums per row
constexpr std::uint32_t TRIANGLE_SOLVE_FACE_PAIR_ROWS  = 2 * TRIANGLE_SOLVE_FACE_ROWS;
constexpr std::uint32_t TRIANGLE_SOLVE_ODD_COLUMNS     = 2; // SFPLOAD address step selecting a face's odd columns of the same rows
// The four block slots of a row group: face 0 even columns, face 0 odd columns, face 1 even columns, face 1 odd columns.
constexpr std::uint32_t TRIANGLE_SOLVE_BLOCK_OFF[TRIANGLE_SOLVE_ROWS_PER_GROUP] = {
    0, TRIANGLE_SOLVE_ODD_COLUMNS, TRIANGLE_SOLVE_FACE_ROWS, TRIANGLE_SOLVE_FACE_ROWS + TRIANGLE_SOLVE_ODD_COLUMNS};
// DEST rows one 32x32 tile occupies, from the same shift set_dst_write_addr uses.
constexpr std::uint32_t TRIANGLE_SOLVE_DEST_TILE_ROWS = 1u << DstTileSizeLog2[DstTileShape::Tile32x32];
static_assert(TRIANGLE_SOLVE_DEST_TILE_ROWS == 2 * TRIANGLE_SOLVE_FACE_PAIR_ROWS, "the solve's DEST layout assumes four 16-row faces per 32x32 tile");

static_assert(TRIANGLE_SOLVE_FACE_DIM % TRIANGLE_SOLVE_ROWS_PER_GROUP == 0, "a row group's columns must not straddle a face boundary");

constexpr std::uint32_t TRIANGLE_SOLVE_IMM16_MASK     = 0xFFFFu; // a 16-bit SFPLOADI immediate
constexpr std::uint32_t TRIANGLE_SOLVE_IMM16_SIGN_BIT = 0x8000u; // its sign bit: bit 15 of a bf16, bit 31 of an fp32 in the upper half

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
    const std::uint32_t face = (row / TRIANGLE_SOLVE_FACE_DIM) * TRIANGLE_SOLVE_FACES_PER_ROW + col / TRIANGLE_SOLVE_FACE_DIM;
    return face * TRIANGLE_SOLVE_FACE_DIM * TRIANGLE_SOLVE_FACE_DIM + (row % TRIANGLE_SOLVE_FACE_DIM) * TRIANGLE_SOLVE_FACE_DIM + col % TRIANGLE_SOLVE_FACE_DIM;
}

/**
 * @brief DEST offset (within a tile) of the first block of a row group.
 *
 * @param group: Row group index, 0..7 (tile rows 4 * group .. 4 * group + 3).
 */
inline constexpr std::uint32_t _triangle_solve_group_base_(const std::uint32_t group)
{
    return (group % TRIANGLE_SOLVE_GROUPS_PER_FACE) * TRIANGLE_SOLVE_ROWS_PER_GROUP + (group / TRIANGLE_SOLVE_GROUPS_PER_FACE) * TRIANGLE_SOLVE_FACE_PAIR_ROWS;
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
 * @brief SFPLOADI into LREG7 with a compile-time XOR applied to the immediate: TT_SFPLOADI(LREG7, MOD0, imm16 ^ FLIP).
 *
 * The immediate fills the zero low half of the instruction word, so the flip is folded into the constant opcode and the RISC
 * builds the word with one XOR; through the TT_SFPLOADI macro the flip would be a separate XOR before the macro's add.
 *
 * @tparam MOD0: SFPLOADI mode, one of sfpi::SFPLOADI_MOD0_*
 * @tparam FLIP: Bits XORed into the immediate, within the low 16 bits
 * @param imm16: The 16-bit immediate; must fit TRIANGLE_SOLVE_IMM16_MASK.
 */
template <std::uint32_t MOD0, std::uint32_t FLIP = 0>
inline void _triangle_solve_sfploadi_lreg7_(const std::uint32_t imm16)
{
    constexpr std::uint32_t OPCODE = TT_OP_SFPLOADI(p_sfpu::LREG7, MOD0, 0);
    static_assert(MOD0 <= 0xF, "SFPLOADI mod0 is a 4-bit field");
    static_assert((OPCODE & TRIANGLE_SOLVE_IMM16_MASK) == 0, "the immediate must fill the zero low half of the SFPLOADI word");
    static_assert(FLIP <= TRIANGLE_SOLVE_IMM16_MASK, "the flip must stay within the 16-bit immediate");
    TT_INSN((OPCODE ^ FLIP) ^ imm16);
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
inline void _triangle_solve_load_l_(const std::uint32_t bits)
{
    static_assert(L_FORMAT == DataFormat::Float32 || L_FORMAT == DataFormat::Float16_b, "the triangle solve reads L as Float32 or Float16_b");
    // Sign bit of the 16-bit immediate: bit 15 of a bf16, and bit 31 of an fp32 once shifted into the upper-half immediate.
    constexpr std::uint32_t SIGN_FLIP = L_NEGATED ? 0 : TRIANGLE_SOLVE_IMM16_SIGN_BIT;
    if constexpr (L_FORMAT == DataFormat::Float32)
    {
        _triangle_solve_sfploadi_lreg7_<sfpi::SFPLOADI_MOD0_UPPER, SIGN_FLIP>(bits >> 16);
        _triangle_solve_sfploadi_lreg7_<sfpi::SFPLOADI_MOD0_LOWER>(bits & TRIANGLE_SOLVE_IMM16_MASK);
    }
    else
    {
        _triangle_solve_sfploadi_lreg7_<sfpi::SFPLOADI_MOD0_FLOATB, SIGN_FLIP>(bits);
    }
}

/**
 * @brief Rank-1 update of the live row group (LREG0..3) from one solved column: LREG r -= L[row0 + r][col] * X[col].
 *
 * X[col] is loaded once into LREG4; the four L entries sit one face row (16 elements) apart in L1 and their reads issue in the
 * shadow of the SFPU instructions. No SFPNOP separates the SFPLOAD from the splats or the first SFPMAD: both loads have
 * 1-cycle results.
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
    _triangle_solve_load_l_<L_FORMAT, L_NEGATED>(b0);
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
 * @brief SFPLOAD the four blocks of a row group into LREG0..3.
 *
 * @param base: DEST offset of the row group's first block.
 */
inline void _triangle_solve_load_blocks_(const std::uint32_t base)
{
    TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[0]);
    TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[1]);
    TT_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[2]);
    TT_SFPLOAD(p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, base + TRIANGLE_SOLVE_BLOCK_OFF[3]);
}

/**
 * @brief Load the four rows of a row group into LREG0..3, row-oriented.
 *
 * SFPTRANSP transposes LREG0..3 and LREG4..7 together. The transpose after the block loads turns the four blocks into
 * four row-oriented registers; the one before them makes the two permutations of LREG4..7 cancel, so LREG5 and LREG6
 * survive the call (the loads overwrite LREG0..3 either way).
 *
 * @param base: DEST offset of the row group's first block.
 */
inline void _triangle_solve_load_group_(const std::uint32_t base)
{
    TTI_SFPTRANSP(0, 0, 0, 0); // all arguments are unused
    _triangle_solve_load_blocks_(base);
    TTI_SFPTRANSP(0, 0, 0, 0); // all arguments are unused
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
    _triangle_solve_load_blocks_(base);
    TTI_SFPTRANSP(0, 0, 0, 0); // all arguments are unused
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
 *       accumulation enabled and ADDR_MOD_7 programmed by @ref _llk_math_eltwise_binary_sfpu_init_. L is read through the
 *       RISC's L1 data cache: invalidate it (invalidate_l1_cache) first if the tile may have been rewritten since it was
 *       last read. Writes LREG0..4 and LREG7.
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

        // Columns of the previous groups, walked one solved group (four columns) at a time. A group's four columns never
        // straddle a face boundary, so they are contiguous in L1; the unrolled body makes each stash slot offset a constant.
        const std::uint32_t row0 = group * TRIANGLE_SOLVE_ROWS_PER_GROUP;
        for (std::uint32_t src = 0; src < group; src++)
        {
            volatile tt_l1_ptr l_elem_t* const l_blk = tile + _triangle_solve_elem_(row0, src * TRIANGLE_SOLVE_ROWS_PER_GROUP);
            const std::uint32_t x_base               = out_base + _triangle_solve_group_base_(src);
#pragma GCC unroll TRIANGLE_SOLVE_ROWS_PER_GROUP
            for (std::uint32_t k = 0; k < TRIANGLE_SOLVE_ROWS_PER_GROUP; k++)
            {
                _triangle_solve_apply_prev_col_<L_FORMAT, L_NEGATED>(l_blk + k, x_base + TRIANGLE_SOLVE_BLOCK_OFF[k]);
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

} // namespace sfpu
} // namespace ckernel
