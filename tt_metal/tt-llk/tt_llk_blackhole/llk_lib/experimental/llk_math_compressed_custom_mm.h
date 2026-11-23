// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "experimental/llk_custom_mm_common.h"
#include "experimental/llk_math_custom_mm.h"

using namespace ckernel;
using namespace ckernel::math;

// CUSTOM_MM
// Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
// in0 tile shape: [{1, 2, 4, 8}, 32] or [32, 32]
// in1 tile shape: [32, 32]
// rt_dim: any integer of 1 or more, as long as the rt_dim * ct_dim output tiles fit in the DEST sync region
// ct_dim: any integer from 1 to 64, with the same DEST bound (custom_mm_tile_rows of the CustomMmDestLayout per
//         output tile); nothing else bounds it
// kt_dim: even number from 2 to 256 (inclusive)
// fidelity: LoFi only
// throttle: not supported
//
// in0 is [rt_dim, kt_dim] tiles and the output [rt_dim, ct_dim] tiles, both in row-major order. The block is done one
// in0 tile-row at a time, each row a pass over the weight metadata into that row's output tiles.

// The first three MVMULs of a weight tile; the execute issues the fourth itself, as its Src clears differ by position.
constexpr std::uint32_t COMPRESSED_CUSTOM_MM_REPLAY_LEN = 3;
// The same for a [32, 32] in0 tile: fifteen of its sixteen MVMULs.
constexpr std::uint32_t COMPRESSED_CUSTOM_MM_FULL_IN0_REPLAY_LEN = 15;

// Address mods a [32, 32] in0 tile adds. An MVMUL multiplies 8 SrcB rows by one SrcA face into 8 DEST rows, so a
// [32, 32] x [32, 32] tile pair is 16 of them. With in0 faces (row, k) at SrcB row 16 * (2 * row + k), in1 faces
// (k, col) at SrcA row 16 * (2 * k + col) and output faces (row, col) at DEST row 16 * (2 * row + col), the walk
// below goes k face, then output face row, then output face column, then the two 8-row halves of the face:
//
//   k  row col half | SrcA SrcB DEST | then
//   0   0   0   0   |   0    0    0  | next half
//   0   0   0   1   |   0    8    8  | next column
//   0   0   1   0   |  16    0   16  | next half
//   0   0   1   1   |  16    8   24  | next face row
//   0   1   0   0   |   0   32   32  | next half
//   0   1   0   1   |   0   40   40  | next column
//   0   1   1   0   |  16   32   48  | next half
//   0   1   1   1   |  16   40   56  | next k face
//   1   ...           32/48 16-56 0-56, the same eight steps one k face on in both sources
//
// SrcA and SrcB counters wrap at 64 rows and DEST at 1024, so a step back is the complement. ADDR_MOD_2 and
// ADDR_MOD_3 of custom_mm_configure_addrmod end the tile and the k-row as they do for a short in0 tile.
inline void _llk_math_compressed_custom_mm_full_in0_addrmod_()
{
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0); // Next half: the lower 8 rows of the same in0 and output faces

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 64 - 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1); // Next column: the next in1 and output face, back to the top half of the in0 face

    addr_mod_t {
        .srca = {.incr = 64 - 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 24, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_4); // Next face row: the in0 face two on, back to the first in1 face of this k face

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 64 - 24, .clr = 0, .cr = 0},
        .dest = {.incr = 1024 - 56, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_5); // Next k face: one face on in SrcA and SrcB, back to the top of the output tile
}

inline void _llk_math_compressed_custom_mm_mop_config_(const bool full_in0)
{
    if (full_in0)
    {
        load_replay_buf(
            ckernel::math::replay_buf_offset,
            COMPRESSED_CUSTOM_MM_FULL_IN0_REPLAY_LEN,
            []
            {
                // Two k faces of eight MVMULs each; the execute issues the very last one
                for (std::uint32_t k_face = 0; k_face < 2; k_face++)
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // row 0, column 0, top half
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // row 0, column 0, bottom half
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // row 0, column 1, top half
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // row 0, column 1, bottom half
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // row 1, column 0, top half
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // row 1, column 0, bottom half
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // row 1, column 1, top half
                    if (k_face == 0)
                    {
                        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0); // row 1, column 1, bottom half
                    }
                }
            });
    }
    else
    {
        load_replay_buf(
            ckernel::math::replay_buf_offset,
            COMPRESSED_CUSTOM_MM_REPLAY_LEN,
            []
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F0 @ F0 => 0
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // F0 @ F1 (F2 if transpose) => 16
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F1 @ F2 (F1 if transpose) => 0
            });
    }
}

/**
 * @brief Configure the math thread for compressed custom_mm: programs address mods and the MVMUL replay buffer.
 *
 * @tparam transpose: Transpose in1 faces during the multiply. Short in0 tiles only.
 * @tparam layout: DEST layout of the output tiles: Default, Dense or Compact, see CustomMmDestLayout. Pack with the
 *                 same layout through @ref _llk_pack_custom_mm_init_. A [32, 32] in0 tile gives [32, 32] output
 *                 tiles, which take the 64 rows of Default.
 * @tparam full_in0: in0 (SrcB) is a [32, 32] tile, values = <true/false>; false is a [{1, 2, 4, 8}, 32] tile.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_compressed_custom_mm_init_, giving it the matching
 *       face count.
 */
template <bool transpose = false, CustomMmDestLayout layout = CustomMmDestLayout::Default, bool full_in0 = false>
inline void _llk_math_compressed_custom_mm_init_()
{
    // There is no split accumulation: finalizing it after a skipped (bfp0) weight tile waits on a SrcA bank the
    // unpacker never fills, which hangs math.
    static_assert(!custom_mm_split_acc(layout), "compressed_custom_mm: the SplitAcc layouts are not supported");
    static_assert(
        !full_in0 || (!transpose && layout == CustomMmDestLayout::Default),
        "compressed_custom_mm: a [32, 32] in0 tile needs the Default layout and no transpose");
    // The same address mods as plain custom_mm: only the per-tile metadata walk in the execute differs.
    custom_mm_configure_addrmod<transpose, layout>();
    if constexpr (full_in0)
    {
        _llk_math_compressed_custom_mm_full_in0_addrmod_();
    }
    _llk_math_compressed_custom_mm_mop_config_(full_in0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Run compressed custom_mm over rt_dim in0 tile-rows of kt_dim k-tiles into DEST, skipping the weight tiles
 *        the metadata marks empty.
 *
 * @tparam layout: Must match the init.
 * @tparam full_in0: Must match the init. A template parameter so that the MVMUL replay is a constant in the tile loop.
 * @param base_address_meta: L1 byte address of the per-tile format metadata.
 * @param dst_index: First DEST output tile, counted in custom_mm_tile_rows of the layout. Output tile (r, c) goes
 *                   to dst_index + r * ct_dim + c.
 * @param kt_dim: Number of k-tiles to accumulate.
 * @param ct_dim: Number of output column tiles, 1 to 64.
 * @param rt_dim: Number of in0 tile-rows, 1 or more. The rt_dim * ct_dim output tiles have to fit in the DEST sync
 *                region: 32 Compact, 16 Dense or 8 Default tiles in a 16-bit SyncHalf DEST, twice that in SyncFull,
 *                half with fp32.
 * @note Call @ref _llk_math_compressed_custom_mm_init_ first.
 * @note Callers that know the DEST sync mode should also assert @ref custom_mm_fits_dest for rt_dim * ct_dim tiles.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_compressed_custom_mm_ with the same rt_dim.
 */
template <CustomMmDestLayout layout = CustomMmDestLayout::Default, bool full_in0 = false>
inline void _llk_math_compressed_custom_mm_(
    const std::uint32_t base_address_meta,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1)
{
    static_assert(!custom_mm_split_acc(layout), "compressed_custom_mm: the SplitAcc layouts are not supported, see the init");
    constexpr std::uint32_t tile_rows     = custom_mm_tile_rows(layout);
    constexpr DstTileShape dst_tile_shape = tile_rows == 16 ? DstTileShape::Tile16x16 : (tile_rows == 32 ? DstTileShape::Tile32x16 : DstTileShape::Tile32x32);
    constexpr std::uint32_t replay_len    = full_in0 ? COMPRESSED_CUSTOM_MM_FULL_IN0_REPLAY_LEN : COMPRESSED_CUSTOM_MM_REPLAY_LEN;
    const std::uint32_t* meta_ptr         = reinterpret_cast<const std::uint32_t*>(base_address_meta);

    // One pass over the weight metadata per in0 tile-row, into that row's ct_dim output tiles
    for (std::uint32_t r = 0; r < rt_dim; r++)
    {
        math::set_dst_write_addr<dst_tile_shape, UnpackDestination::SrcRegs>(dst_index + r * ct_dim);

        std::uint32_t index      = 0;
        std::uint32_t meta_index = 0;
        std::uint32_t meta       = meta_ptr[meta_index] >> 3;

        for (std::uint32_t i = 0; i < kt_dim; i++)
        {
            for (std::uint32_t j = 0; j < ct_dim - 1; j++)
            {
                std::uint32_t fmt = meta & 0x3;
                if (fmt != 0)
                {
                    lltt::replay(ckernel::math::replay_buf_offset, replay_len);
                    TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
                }
                else
                {
                    TTI_ADDRMOD_ONLY(ADDR_MOD_2);
                }
                index++;
                if (index == 10)
                {
                    index = 0;
                    meta_index++;
                    meta = meta_ptr[meta_index] >> 3;
                }
                else
                {
                    meta >>= 3;
                }
            }
            std::uint32_t fmt = meta & 0x3;
            if (fmt != 0)
            {
                lltt::replay(ckernel::math::replay_buf_offset, replay_len);
                TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);
            }
            else
            {
                // Wait for math to drain first, so SRCB_VLD is tested on the bank the previous MVMUL's CLR_AB flipped to
                TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::MATH | ckernel::p_stall::SRCB_VLD);
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
            }
            index++;
            if (index == 10)
            {
                index = 0;
                meta_index++;
                meta = meta_ptr[meta_index] >> 3;
            }
            else
            {
                meta >>= 3;
            }
        }
    }
}
