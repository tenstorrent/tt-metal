// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu_types.h"

template <typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Callable&& sfpu_func, std::uint32_t dst_index_in, std::uint32_t dst_index_out, int vector_mode = static_cast<int>(VectorMode::RC), Args&&... args)
{
    LLK_ASSERT((dst_index_in < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in exceeds max dest tiles");
    LLK_ASSERT((dst_index_out < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_out exceeds max dest tiles");

    // Set base to 0: callbacks address tiles absolutely via dst_reg[idx * SFP_DST_TILE_ROWS].
    // Follows the binary SFPU pattern (_llk_math_eltwise_binary_sfpu_params_).
    _llk_math_eltwise_unary_sfpu_start_<DST_SYNC_MODE>(0);

    VectorMode mode = static_cast<VectorMode>(vector_mode);

    if (mode == VectorMode::R)
    {
        // Do a row vector, Face0 + Face1 -- first iteration (first row)
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in, dst_index_out, std::forward<Args>(args)...);
            // Move to the next face
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
        // Skip next two faces
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
    }
    else if (mode == VectorMode::C)
    {
        // Do a column vector, Face0 + Face2 -- All iterations for full face
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in, dst_index_out, std::forward<Args>(args)...);
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    }
    else if (mode == VectorMode::RC)
    {
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in, dst_index_out, std::forward<Args>(args)...);
            // Move to the next face
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    }
    else
    {
        std::forward<Callable>(sfpu_func)(dst_index_in, dst_index_out, std::forward<Args>(args)...);
    }
    _llk_math_eltwise_unary_sfpu_done_();
}
