// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

namespace block_max8
{

/**
 * @brief Map a four-row logical band to physical DST rows relative to the dispatcher base.
 *
 * Faces F0/F1 precede F2/F3 in DST. This returns the left-face address;
 * adding FACE_R_DIM selects the right face. A BF16 SFPLOAD covers four rows,
 * while sfpi::dst_reg indexes physical addresses in two-row units.
 * @param band Four-row logical band, starting at zero.
 */
inline std::uint32_t face_row_address(std::uint32_t band)
{
    const std::uint32_t row = band * 4;
    return (row / FACE_R_DIM) * 2 * FACE_R_DIM + row % FACE_R_DIM;
}

/**
 * @brief Rotate values right across the eight instances, independently per physical lane.
 * @tparam STEPS Number of one-instance rotations; the SFPI compiler schedules the shuffles.
 */
template <int STEPS>
inline sfpi::vFloat rotate_right(sfpi::vFloat value)
{
#pragma GCC unroll 8
    for (int step = 0; step < STEPS; ++step)
    {
        value = sfpi::subvec_shflror1(value);
    }
    return value;
}

/**
 * @brief Load one parity of a four-row face band and replace invalid scores with negative infinity.
 *
 * For BF16, vector element = 8 * physical_lane + instance, and vConstTileId
 * is twice that element. Its upper bits select the row; its low bits select
 * the even column, with PARITY selecting even or odd scores. The logical score
 * index is row * TILE_C_DIM + column. Indices >= valid_scores are invalid.
 * @tparam FACE Left or right face within a face pair (0 or 1).
 * @tparam PARITY Even or odd face columns (0 or 1).
 * @param band Four-row logical band.
 * @param valid_scores Number of valid scores in the tile's row-major prefix.
 */
template <std::uint32_t FACE, std::uint32_t PARITY>
inline sfpi::vFloat load_masked(std::uint32_t band, std::uint32_t valid_scores)
{
    const std::uint32_t address = face_row_address(band) + FACE * FACE_R_DIM + PARITY * 2;
    sfpi::vFloat value          = sfpi::dst_reg[address / 2];
    sfpi::vInt row_in_band      = sfpi::vConstTileId >> __builtin_ctz(FACE_C_DIM);
    sfpi::vInt column           = (sfpi::vConstTileId & static_cast<int>(FACE_C_DIM - 2)) + FACE * FACE_C_DIM + PARITY;
    sfpi::vInt index            = ((row_in_band + band * 4) << __builtin_ctz(TILE_C_DIM)) + column;
    v_if (index >= static_cast<int>(valid_scores))
    {
        value = -__builtin_inff();
    }
    v_endif;
    return value;
}

/**
 * @brief Retain the pairwise maximum in EVEN and use ODD as scratch afterward.
 *
 * SFPSWAP writes the maximum to its first source register and the minimum
 * to its second. SFPNOP supplies Blackhole's required idle SFPU cycle.
 */
template <std::uint32_t EVEN, std::uint32_t ODD>
inline void max_pair()
{
    TTI_SFPSWAP(0, EVEN, ODD, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPNOP;
}

/**
 * @brief Apply pairwise max to LREG pairs 0/1, 2/3, 4/5, and 6/7.
 *
 * These pairs represent two four-row bands from each of the left and right faces.
 */
inline void max_four_pairs()
{
    max_pair<0, 1>();
    max_pair<2, 3>();
    max_pair<4, 5>();
    max_pair<6, 7>();
}

/**
 * @brief Rotate SRC right by one instance into DST without mixing physical lanes.
 *
 * The source may equal the destination. SFPNOP supplies Blackhole's required
 * idle SFPU cycle after the shuffle.
 */
template <std::uint32_t SRC, std::uint32_t DST>
inline void shuffle_right_one()
{
    TTI_SFPSHFT2(0, SRC, DST, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
}

/**
 * @brief Rotate all four independent reductions into their odd-numbered scratch LREGs.
 * @tparam FROM_MAX Select even-numbered maxima as sources; otherwise rotate scratch in place.
 */
template <bool FROM_MAX>
inline void shuffle_four_pairs()
{
    shuffle_right_one<FROM_MAX ? 0 : 1, 1>();
    shuffle_right_one<FROM_MAX ? 2 : 3, 3>();
    shuffle_right_one<FROM_MAX ? 4 : 5, 5>();
    shuffle_right_one<FROM_MAX ? 6 : 7, 7>();
}

/**
 * @brief Reduce eight rows across a face pair in place using all eight working LREGs.
 *
 * LREG0/1 and LREG2/3 load even/odd columns from the first four rows of the
 * left/right faces; LREG4/5 and LREG6/7 load the next four rows. All eight
 * loads precede stores, so overwriting the input is safe. Pairwise max,
 * rotate-by-one/max, and rotate-by-two/max produce independent maxima of eight.
 * There is no cross-face max. Instances 3 and 7 contain the useful results;
 * even-column stores place them at face columns 6 and 14.
 *
 * The scalar, non-inlined boundary isolates explicit LREG0..7 use from the
 * SFPI register allocator used by masking and compaction.
 * @param address Left-face physical DST row address of the eight-row group.
 */
__attribute__((noinline)) inline void reduce_eight_rows(std::uint32_t address)
{
    TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, address);
    TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, address + 2);
    TT_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, address + FACE_R_DIM);
    TT_SFPLOAD(p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, address + FACE_R_DIM + 2);
    TT_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, address + 4);
    TT_SFPLOAD(p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, address + 6);
    TT_SFPLOAD(p_sfpu::LREG6, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, address + FACE_R_DIM + 4);
    TT_SFPLOAD(p_sfpu::LREG7, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, address + FACE_R_DIM + 6);

    max_four_pairs();
    shuffle_four_pairs<true>();
    max_four_pairs();
    shuffle_four_pairs<true>();
    shuffle_four_pairs<false>();
    max_four_pairs();

    TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_FP16B, ADDR_MOD_7, address);
    TT_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_FP16B, ADDR_MOD_7, address + FACE_R_DIM);
    TT_SFPSTORE(p_sfpu::LREG4, sfpi::SFPSTORE_MOD0_FMT_FP16B, ADDR_MOD_7, address + 4);
    TT_SFPSTORE(p_sfpu::LREG6, sfpi::SFPSTORE_MOD0_FMT_FP16B, ADDR_MOD_7, address + FACE_R_DIM + 4);
}

/**
 * @brief Gather one block maximum from each face into instances 0 and 1.
 *
 * The first/second block resides in instance 3/7, corresponding to even face
 * column 6/14. The two gathered values form one parity of the final four-score
 * row; other instances are zeroed for the later bitwise merge.
 * @tparam BLOCK Select the first or second eight-score block within each face.
 * @param src_address Left-face physical DST row address for four logical rows.
 */
template <std::uint32_t BLOCK>
inline sfpi::vFloat gather_face_blocks(std::uint32_t src_address)
{
    constexpr std::uint32_t instance = 3 + 4 * BLOCK;
    sfpi::vFloat left                = rotate_right<8 - instance>(sfpi::dst_reg[src_address / 2]);
    sfpi::vFloat right               = rotate_right<9 - instance>(sfpi::dst_reg[(src_address + FACE_R_DIM) / 2]);
    sfpi::vFloat blocks              = 0.0f;
    v_if ((sfpi::vConstTileId & static_cast<int>(FACE_C_DIM - 2)) == 0)
    {
        blocks = left;
    }
    v_endif;
    v_if ((sfpi::vConstTileId & static_cast<int>(FACE_C_DIM - 2)) == 2)
    {
        blocks = right;
    }
    v_endif;
    return blocks;
}

/**
 * @brief Compact one output parity from four logical rows into a selected physical DST row.
 *
 * Gather two block scores from the strided face results for each input row.
 * Transpose four copies to broadcast each
 * input row, rotate into disjoint instance pairs, and merge with bitwise OR.
 * This preserves signed zero and infinities without arithmetic on padding.
 *
 * SFPTRANSP affects both LREG0..3 and LREG4..7. The scalar, non-inlined boundary
 * prevents unrelated caller vectors from remaining live across the transpose.
 * @tparam BLOCK Select the first or second block within each face.
 * @param src_address Left-face physical source address for four logical rows.
 * @param dst_address Physical output address, including output parity.
 * @param output_row Physical lane to store, selecting one row of the four-row output group.
 */
template <std::uint32_t BLOCK>
__attribute__((noinline)) inline void compact_row_band(std::uint32_t src_address, std::uint32_t dst_address, std::uint32_t output_row)
{
    sfpi::vFloat row0 = gather_face_blocks<BLOCK>(src_address);
    sfpi::vFloat row1 = row0, row2 = row0, row3 = row0;
    sfpi::subvec_transp(row0, row1, row2, row3);
    row1 = rotate_right<2>(row1);
    row2 = rotate_right<4>(row2);
    row3 = rotate_right<6>(row3);
    sfpi::vFloat packed =
        sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(row0) | sfpi::as<sfpi::vUInt>(row1) | sfpi::as<sfpi::vUInt>(row2) | sfpi::as<sfpi::vUInt>(row3));
    v_if ((sfpi::vConstTileId >> __builtin_ctz(FACE_C_DIM)) == static_cast<int>(output_row))
    {
        sfpi::dst_reg[dst_address / 2] = packed;
    }
    v_endif;
}

} // namespace block_max8

/**
 * @brief Mask scores outside the valid row-major prefix in the current DST tile.
 *
 * Each load/store touches one parity of four face rows. Each vector is loaded
 * before its own positions are overwritten, preserving unread inputs. Invalid
 * scores become negative infinity before any reduction. All addresses are
 * relative to the tile selected by the standard unary dispatcher.
 * @param valid_scores Number of valid scores, from zero through TILE_R_DIM * TILE_C_DIM.
 */
inline void _calculate_block_max8_mask_(std::uint32_t valid_scores)
{
    for (std::uint32_t band = 0; band < TILE_R_DIM / 4; ++band)
    {
        const std::uint32_t address                   = block_max8::face_row_address(band);
        sfpi::dst_reg[address / 2]                    = block_max8::load_masked<0, 0>(band, valid_scores);
        sfpi::dst_reg[(address + 2) / 2]              = block_max8::load_masked<0, 1>(band, valid_scores);
        sfpi::dst_reg[(address + FACE_R_DIM) / 2]     = block_max8::load_masked<1, 0>(band, valid_scores);
        sfpi::dst_reg[(address + FACE_R_DIM + 2) / 2] = block_max8::load_masked<1, 1>(band, valid_scores);
    }
}

/**
 * @brief Replace one BF16 DST tile with 128 consecutive maxima of eight scores.
 *
 * The standard unary dispatcher anchors DST at the caller's tile index; all
 * addresses here are relative to that tile. A partial tile is masked in place
 * before the eight-register reduction. Full tiles skip masking entirely.
 *
 * Reduction loads each complete eight-row group before overwriting it.
 * Compaction processes bands in ascending order and writes odd output columns
 * before even ones: the first band's even inputs must survive both gathers.
 * Later stores overwrite only consumed rows. The first eight physical DST
 * rows contain the result; the rest of the tile is scratch and is not preserved.
 * @param valid_scores Number of valid scores in the input's row-major prefix.
 */
inline void _calculate_block_max8_(std::uint32_t valid_scores)
{
    if (valid_scores < TILE_R_DIM * TILE_C_DIM)
    {
        _calculate_block_max8_mask_(valid_scores);
    }
    for (std::uint32_t band = 0; band < TILE_R_DIM / 4; band += 2)
    {
        block_max8::reduce_eight_rows(block_max8::face_row_address(band));
    }
    for (std::uint32_t band = 0; band < TILE_R_DIM / 4; ++band)
    {
        const std::uint32_t address = block_max8::face_row_address(band);
        const std::uint32_t packed  = (band / 4) * 4;
        block_max8::compact_row_band<1>(address, packed + 2, band % 4);
        block_max8::compact_row_band<0>(address, packed, band % 4);
    }
}

} // namespace ckernel::sfpu
