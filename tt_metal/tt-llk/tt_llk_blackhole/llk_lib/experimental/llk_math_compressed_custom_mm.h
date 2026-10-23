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
// in0 tile shape: [{1, 2, 4, 8}, 32]
// in1 tile shape: [32, 32]
// rt_dim: 1
// ct_dim: any integer from 1 to 64, as long as the ct_dim output tiles fit in the DEST sync region
//         (custom_mm_tile_rows of the CustomMmDestLayout each); nothing else bounds it
// kt_dim: even number from 2 to 256 (inclusive)
// fidelity: LoFi only
// throttle: not supported

inline void _llk_math_compressed_custom_mm_mop_config_(const std::uint32_t operandB_face_r_dim)
{
    const std::uint32_t replay_buf_len = operandB_face_r_dim == 8 ? 11 : 9;

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        [operandB_face_r_dim]
        {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F0 @ F0 => 0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // F0 @ F1 (F2 if transpose) => 16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F1 @ F2 (F1 if transpose) => 0 (8 if split_acc)

            // Finalization phase
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // F1 @ F3 => 16 (24 if split_acc) (clear none,
                                                             // as both SrcA and SrcB banks are required for finalization)

            // We will exploit AddDst for finalization to calculate:
            // Dst = partial (in dest) + (split partial (moved to SrcB) + 0 (zeroed SrcA))
            // This way we avoid additional set of moves and move only one set of partials to SrcB
            // We have to move them to SrcB as we have to zero out the other Src register
            // and that can only be SrcA as it is not reused
            // Since we are going to keep reusing SrcB,
            // we have to move partials in lower half of SrcB as upper half contains inputs that are being reused

            // Move the 4 rows of the F0 split partial to rows 32+ of SrcB for finalization
            TTI_MOVD2B(0, 32, ADDR_MOD_5, p_movd2a::MOV_4_ROWS, 0 + 8);
            // Move the 4 rows of the F1 split partial to rows 48+ of SrcB for finalization
            TTI_MOVD2B(0, 16, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 16 + 8);

            // Move lower 4 rows if they exist
            if (operandB_face_r_dim == 8)
            {
                TTI_MOVD2B(0, 0 + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 0 + 8 + 4);
                TTI_MOVD2B(0, 16 + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 16 + 8 + 4);
            }

            // Zero SrcA as it is not reused and we need to add zero to split partials in SrcB
            TTI_ZEROSRC(0, 1, 0, 1);

            // F0 partial in dest + F0 split partial in SrcB (row 32) => final F0 in dest
            TTI_ELWADD(0, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_6, 0);
            // F1 partial in dest + F1 split partial in SrcB (row 48) => final F1 in dest (clear only SrcA as B is reused)
            TTI_ELWADD(1, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
        });
}

/**
 * @brief Step past a skipped weight tile that is not last in its row: what ADDR_MOD_2 does after a tile's MVMULs.
 *
 * Moves the Dst counter and its carriage return one output tile on and puts the SrcA and SrcB counters back to 0,
 * without touching DEST or the Src valids. An address-mode-only ZEROACC cannot do this on Blackhole: CLR_16 clears a
 * 16-row face of DEST whatever its row operand, even 0xff, and that face can sit in the packer's half.
 *
 * @tparam tile_rows: DEST rows per output tile, custom_mm_tile_rows of the layout.
 */
template <std::uint32_t tile_rows>
inline void _llk_math_compressed_custom_mm_skip_tile_()
{
    // The Dst field of SETRWC and INCRWC holds at most 15, so step 8 rows at a time; CR_D moves the carriage return
    // too, as ADDR_MOD_2's dest.cr does.
    constexpr std::uint32_t step = 8;
    static_assert(tile_rows % step == 0, "tile_rows must be a multiple of the step");
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, step, 0, 0, p_setrwc::SET_ABD);
    for (std::uint32_t rows = step; rows < tile_rows; rows += step)
    {
        TTI_INCRWC(p_setrwc::CR_D, step, 0, 0);
    }
}

/**
 * @brief Configure the math thread for compressed custom_mm: programs address mods and the MVMUL replay buffer.
 *
 * @tparam transpose: Transpose in1 faces during the multiply.
 * @tparam layout: DEST layout of the output tiles: Default, Dense or Compact, see CustomMmDestLayout. Pack with the
 *                 same layout through @ref _llk_pack_custom_mm_init_.
 * @param operandB_face_r_dim: Face row count of in0 (SrcB), one of {1, 2, 4, 8}.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_compressed_custom_mm_init_.
 */
template <bool transpose = false, CustomMmDestLayout layout = CustomMmDestLayout::Default>
inline void _llk_math_compressed_custom_mm_init_(const std::uint32_t operandB_face_r_dim)
{
    // The finalization of a skipped (bfp0) weight tile waits on a SrcA bank the unpacker never fills, which hangs math,
    // so the SplitAcc layouts, which finalize, are not supported.
    static_assert(!custom_mm_split_acc(layout), "compressed_custom_mm: the SplitAcc layouts are not supported");
    // The same address mods as plain custom_mm: only the per-tile metadata walk in the execute differs.
    custom_mm_configure_addrmod<transpose, layout>();
    _llk_math_compressed_custom_mm_mop_config_(operandB_face_r_dim);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Run compressed custom_mm over kt_dim k-tiles into DEST, skipping the weight tiles the metadata marks empty.
 *
 * @tparam finalize: Must be false: finalization merges split partials, and no supported layout has them.
 * @tparam layout: Must match the init.
 * @param base_address_meta: L1 byte address of the per-tile format metadata.
 * @param operandB_face_r_dim: Face row count of in0 (SrcB), one of {1, 2, 4, 8}.
 * @param dst_index: First DEST output tile, counted in custom_mm_tile_rows of the layout.
 * @param kt_dim: Number of k-tiles to accumulate.
 * @param ct_dim: Number of output column tiles, 1 to 64, as long as they fit in the DEST sync region: 32 Compact,
 *                16 Dense or 8 Default tiles in a 16-bit SyncHalf DEST, twice that in SyncFull, half with fp32.
 * @note Call @ref _llk_math_compressed_custom_mm_init_ first.
 * @note Callers that know the DEST sync mode should also assert @ref custom_mm_fits_dest.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_compressed_custom_mm_.
 */
template <bool finalize = false, CustomMmDestLayout layout = CustomMmDestLayout::Default>
inline void _llk_math_compressed_custom_mm_(
    const std::uint32_t base_address_meta,
    const std::uint32_t operandB_face_r_dim,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1)
{
    const std::uint32_t replay_buf_len = operandB_face_r_dim == 8 ? 11 : 9;
    static_assert(!custom_mm_split_acc(layout), "compressed_custom_mm: the SplitAcc layouts are not supported, see the init");
    static_assert(!finalize, "compressed_custom_mm: finalize merges split partials, and no supported layout has them");
    constexpr std::uint32_t tile_rows     = custom_mm_tile_rows(layout);
    constexpr DstTileShape dst_tile_shape = tile_rows == 16 ? DstTileShape::Tile16x16 : (tile_rows == 32 ? DstTileShape::Tile32x16 : DstTileShape::Tile32x32);
    math::set_dst_write_addr<dst_tile_shape, UnpackDestination::SrcRegs>(dst_index);

    const std::uint32_t iterations = finalize ? kt_dim - 1 : kt_dim;
    const std::uint32_t* meta_ptr  = reinterpret_cast<const std::uint32_t*>(base_address_meta);
    std::uint32_t index            = 0;
    std::uint32_t meta_index       = 0;
    std::uint32_t meta             = meta_ptr[meta_index] >> 3;

    for (std::uint32_t i = 0; i < iterations; i++)
    {
        for (std::uint32_t j = 0; j < ct_dim - 1; j++)
        {
            std::uint32_t fmt = meta & 0x3;
            if (fmt != 0)
            {
                lltt::replay(ckernel::math::replay_buf_offset, 3);
                TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
            }
            else
            {
                _llk_math_compressed_custom_mm_skip_tile_<custom_mm_tile_rows(layout)>();
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
            lltt::replay(ckernel::math::replay_buf_offset, 3);
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

    if constexpr (finalize)
    {
        for (std::uint32_t j = 0; j < ct_dim - 1; j++)
        {
            std::uint32_t fmt = meta & 0x3;
            if (fmt != 0)
            {
                lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
            }
            else
            {
                TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::SRCA_VLD | 0);
                lltt::replay(ckernel::math::replay_buf_offset + 4, replay_buf_len - 4);
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
            lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len - 1);
            TTI_ELWADD(3, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
        }
        else
        {
            TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::SRCA_VLD | ckernel::p_stall::SRCB_VLD);
            lltt::replay(ckernel::math::replay_buf_offset + 4, replay_buf_len - 5);
            TTI_ELWADD(3, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
        }
    }
}
