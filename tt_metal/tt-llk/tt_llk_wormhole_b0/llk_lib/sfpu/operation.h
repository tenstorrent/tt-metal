// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"
#include "sfpu/common.h"
#include "sfpu/traverse.h"
#include "tensor_shape.h"

namespace llk::sfpu
{

class Operation
{
protected:
    /**
     * @brief Assert Dest tile @p index is within the Dest acquired for @p Sync.
     *
     * @tparam Sync: Dest sync mode, values = <SyncHalf/SyncFull>
     * @param index: Dest tile to check
     */
    template <DstSync Sync>
    static void assert_dest_index_valid(const std::uint32_t index)
    {
        LLK_ASSERT((index < get_dest_max_tiles_rt<Sync, DstTileShape::Tile32x32>()), "Dest tile index is outside the acquired Dest");
    }

    /**
     * @brief Assert the @p tiles Dest tiles starting at @p first are all within the Dest acquired for @p Sync.
     *
     * @tparam Sync: Dest sync mode, values = <SyncHalf/SyncFull>
     * @param first: First Dest tile of the run
     * @param tiles: Number of tiles in the run, at least 1
     */
    template <DstSync Sync>
    static void assert_dest_range_valid(const std::uint32_t first, const std::uint32_t tiles)
    {
        // Compared as a count left after first, so a large tiles cannot wrap first + tiles - 1 back into range.
        LLK_ASSERT(
            (tiles > 0) && (first < get_dest_max_tiles_rt<Sync, DstTileShape::Tile32x32>()) &&
                (tiles <= get_dest_max_tiles_rt<Sync, DstTileShape::Tile32x32>() - first),
            "Dest tile range is empty or extends outside the acquired Dest");
    }

    /**
     * @brief Assert Dest tiles @p a and @p b are different tiles.
     *
     * @param a: Dest tile
     * @param b: Dest tile
     */
    static void assert_distinct(const std::uint32_t a, const std::uint32_t b)
    {
        LLK_ASSERT(a != b, "Dest tiles must be distinct");
    }

    /**
     * @brief Fail to compile unless @p Shape fits the face walk. See @ref is_shape_walkable.
     *
     * @tparam Shape: Region the op walks
     */
    template <const TensorShape& Shape>
    static constexpr void assert_shape_walkable()
    {
        static_assert(is_shape_walkable(Shape), "Shape must fit in a tile: faces of 1..16 x 1..16 in a grid of 1..2 x 1..2");
    }

    /**
     * @brief Fail to compile if a padding-sensitive op would walk lanes outside @p Shape. See @ref is_walk_agnostic.
     *
     * @tparam Shape: Region the op walks
     * @tparam Agnostic: Whether the op may compute on lanes outside @p Shape
     */
    template <const TensorShape& Shape, PaddingAgnostic Agnostic>
    static constexpr void assert_walk_agnostic()
    {
        static_assert(Agnostic == PaddingAgnostic::Yes || is_walk_agnostic(Shape), "a padding-sensitive op cannot walk Shape: the walk touches padding");
    }
};

} // namespace llk::sfpu
