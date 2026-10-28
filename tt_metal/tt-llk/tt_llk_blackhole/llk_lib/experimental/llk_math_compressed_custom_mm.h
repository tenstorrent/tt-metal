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

// The first three MVMULs of a weight tile; the execute issues the fourth itself, as its Src clears differ by position.
constexpr std::uint32_t COMPRESSED_CUSTOM_MM_REPLAY_LEN = 3;

inline void _llk_math_compressed_custom_mm_mop_config_()
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

/**
 * @brief Configure the math thread for compressed custom_mm: programs address mods and the MVMUL replay buffer.
 *
 * @tparam transpose: Transpose in1 faces during the multiply.
 * @tparam layout: DEST layout of the output tiles: Default, Dense or Compact, see CustomMmDestLayout. Pack with the
 *                 same layout through @ref _llk_pack_custom_mm_init_.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_compressed_custom_mm_init_.
 */
template <bool transpose = false, CustomMmDestLayout layout = CustomMmDestLayout::Default>
inline void _llk_math_compressed_custom_mm_init_()
{
    // There is no split accumulation: finalizing it after a skipped (bfp0) weight tile waits on a SrcA bank the
    // unpacker never fills, which hangs math.
    static_assert(!custom_mm_split_acc(layout), "compressed_custom_mm: the SplitAcc layouts are not supported");
    // The same address mods as plain custom_mm: only the per-tile metadata walk in the execute differs.
    custom_mm_configure_addrmod<transpose, layout>();
    _llk_math_compressed_custom_mm_mop_config_();

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Run compressed custom_mm over kt_dim k-tiles into DEST, skipping the weight tiles the metadata marks empty.
 *
 * @tparam layout: Must match the init.
 * @param base_address_meta: L1 byte address of the per-tile format metadata.
 * @param dst_index: First DEST output tile, counted in custom_mm_tile_rows of the layout.
 * @param kt_dim: Number of k-tiles to accumulate.
 * @param ct_dim: Number of output column tiles, 1 to 64, as long as they fit in the DEST sync region: 32 Compact,
 *                16 Dense or 8 Default tiles in a 16-bit SyncHalf DEST, twice that in SyncFull, half with fp32.
 * @note Call @ref _llk_math_compressed_custom_mm_init_ first.
 * @note Callers that know the DEST sync mode should also assert @ref custom_mm_fits_dest.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_compressed_custom_mm_.
 */
template <CustomMmDestLayout layout = CustomMmDestLayout::Default>
inline void _llk_math_compressed_custom_mm_(
    const std::uint32_t base_address_meta, const std::uint32_t dst_index, const std::uint32_t kt_dim, const std::uint32_t ct_dim = 1)
{
    static_assert(!custom_mm_split_acc(layout), "compressed_custom_mm: the SplitAcc layouts are not supported, see the init");
    constexpr std::uint32_t tile_rows     = custom_mm_tile_rows(layout);
    constexpr DstTileShape dst_tile_shape = tile_rows == 16 ? DstTileShape::Tile16x16 : (tile_rows == 32 ? DstTileShape::Tile32x16 : DstTileShape::Tile32x32);
    math::set_dst_write_addr<dst_tile_shape, UnpackDestination::SrcRegs>(dst_index);

    const std::uint32_t* meta_ptr = reinterpret_cast<const std::uint32_t*>(base_address_meta);
    std::uint32_t index           = 0;
    std::uint32_t meta_index      = 0;
    std::uint32_t meta            = meta_ptr[meta_index] >> 3;

    for (std::uint32_t i = 0; i < kt_dim; i++)
    {
        for (std::uint32_t j = 0; j < ct_dim - 1; j++)
        {
            std::uint32_t fmt = meta & 0x3;
            if (fmt != 0)
            {
                lltt::replay(ckernel::math::replay_buf_offset, COMPRESSED_CUSTOM_MM_REPLAY_LEN);
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
            lltt::replay(ckernel::math::replay_buf_offset, COMPRESSED_CUSTOM_MM_REPLAY_LEN);
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
