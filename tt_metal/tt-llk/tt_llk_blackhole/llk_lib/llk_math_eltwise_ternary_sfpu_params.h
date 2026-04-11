// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "llk_sfpu_types.h"

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_in2,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    LLK_ASSERT((dst_index_in0 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in0 exceeds max dest tiles");
    LLK_ASSERT((dst_index_in1 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in1 exceeds max dest tiles");
    LLK_ASSERT((dst_index_in2 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in2 exceeds max dest tiles");
    LLK_ASSERT((dst_index_out < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_out exceeds max dest tiles");

    _llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(0); // Reuse same sync primitive

    if (vector_mode == static_cast<int>(VectorMode::R))
    {
        // Row vector - Face0 + Face1
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // repeat 2x
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
        // Skip next 2 faces
        for (int i = 0; i < 4; ++i)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }
    else if (vector_mode == static_cast<int>(VectorMode::C))
    {
        // Column vector - Face0 + Face2
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
            for (int i = 0; i < 4; ++i)
            {
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    else if (vector_mode == static_cast<int>(VectorMode::RC))
    {
        // All 4 faces
        for (int face = 0; face < 4; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }
    else
    {
        // Default: single face pass-through
        std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
    }
    _llk_math_eltwise_ternary_sfpu_done_(); // Finalize
}
