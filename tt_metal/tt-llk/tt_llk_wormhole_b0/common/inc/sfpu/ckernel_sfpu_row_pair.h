// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Two-row SFPU loop bodies: one iteration reads and writes Dest rows 0 and 1, and the second row's
// store also advances Dest past both rows, so the body needs no separate TTINCRWC. That store names
// ADDR_MOD_2; unary SFPU sets addr_mod_base, so it reaches physical slot 6, which
// _init_sfpu_row_pair_() programs with a Dest increment of the two rows' counter steps.
//
// A throttled matmul (MM_THROTTLE > 0, the compute-kernel define ttnn's throttle_mm_perf sets) programs
// ADDR_MOD_6 itself, and the fused-activation matmul/conv kernels run the activation's init once, before
// matmul_block_init. Under that define the body advances Dest explicitly and the init leaves slot 6 alone.
#if defined(MM_THROTTLE) && MM_THROTTLE > 0
constexpr bool SFPU_ROW_PAIR_STORE_ADVANCES_DEST = false;
#else
constexpr bool SFPU_ROW_PAIR_STORE_ADVANCES_DEST = true;
#endif

// Dest counter steps per loop body: one sfpi row is two counter steps.
constexpr std::int16_t SFPU_ROW_PAIR_DEST_INCR = 4;

/**
 * Programs the Dest increment the second-row store of a two-row body uses.
 * @note Call from the op's init; nothing may rewrite ADDR_MOD_6 between that init and the op's calls.
 */
inline void _init_sfpu_row_pair_()
{
    if constexpr (SFPU_ROW_PAIR_STORE_ADVANCES_DEST)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = SFPU_ROW_PAIR_DEST_INCR},
        }
            .set(ADDR_MOD_6);
    }
}

/**
 * Stores the second row of a two-row body; unless MM_THROTTLE is set, the store also advances Dest past
 * both rows. May sit inside a v_if: a store with every lane disabled still applies its address mode.
 */
sfpi_inline void _sfpu_row_pair_store_second_(const sfpi::vFloat value)
{
    if constexpr (SFPU_ROW_PAIR_STORE_ADVANCES_DEST)
    {
        sfpi::dst_reg[1].mode(ADDR_MOD_2) = value;
    }
    else
    {
        sfpi::dst_reg[1] = value;
    }
}

/**
 * Ends a two-row body: advances Dest past both rows when the second store did not.
 */
sfpi_inline void _sfpu_row_pair_end_()
{
    if constexpr (!SFPU_ROW_PAIR_STORE_ADVANCES_DEST)
    {
        sfpi::dst_reg += 2;
    }
}

} // namespace ckernel::sfpu
