// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

/**
 * @brief Fill the current full BF16 DST tile with exact negative-zero values.
 *
 * Invoke once through the unary SFPU dispatcher with VectorMode::RC_custom,
 * FP32 destination accumulation disabled, and all lanes enabled. The dispatcher
 * selects the tile and owns synchronization; this helper writes all four faces
 * relative to that base without advancing it or touching neighboring tiles.
 *
 * Materialize the sign bit directly to avoid arithmetic or unpack/datacopy paths
 * that may canonicalize negative zero. Both even and odd columns are written in
 * each four-row BF16 load/store group. Subsequent packing may still canonicalize
 * zero signs, so bit-sensitive consumers must operate on the DST values directly.
 */
inline void _calculate_fill_negative_zero_()
{
    sfpi::vFloat negative_zero = sfpi::as<sfpi::vFloat>(sfpi::vUInt(0x80000000u));
    for (std::uint32_t row = 0; row < TILE_NUM_FACES * FACE_R_DIM; row += 4)
    {
        sfpi::dst_reg[row / 2]     = negative_zero;
        sfpi::dst_reg[row / 2 + 1] = negative_zero;
    }
}

} // namespace ckernel::sfpu
