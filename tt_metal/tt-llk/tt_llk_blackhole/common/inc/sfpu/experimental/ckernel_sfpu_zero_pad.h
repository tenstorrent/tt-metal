// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// SFPU kernel: zero padding rows in a dest register tile.
//
// After squaring, the last tile may contain garbage in the padding region.
// This kernel zeros those rows so the subsequent reduction is correct.
// Each SFPU iteration processes 32 bf16 elements (64 bytes) in 1 cycle.

#pragma once

#include "ckernel.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// Zero rows [VALID_ROWS, TOTAL_ROWS) in the current dest tile.
// VALID_ROWS: SFPU rows containing real squared data.
// TOTAL_ROWS: total SFPU rows per tile (16 for HALF, 32 for FULL).
template <bool is_fp32_dest_acc_en, int VALID_ROWS, int TOTAL_ROWS>
inline void _zero_pad_tile_()
{
    // Advance past valid rows without writing.
    for (int d = 0; d < VALID_ROWS; d++)
    {
        sfpi::dst_reg++;
    }
    // Zero the padding rows.
    sfpi::vFloat zero = 0.0f;
    for (int d = VALID_ROWS; d < TOTAL_ROWS; d++)
    {
        sfpi::dst_reg[0] = zero;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
