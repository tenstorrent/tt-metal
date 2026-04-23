// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-08_abs_quasar_2f52d870
#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{
// Calculates ABS for number of rows of output SFPU ops (Quasar = 2 rows).
//
// Template dispatch on the Dest-register data format. For Int8, Dest stores
// signed 8-bit data in sign+magnitude layout (see Confluence "Dest storage
// formats", page 80674824), so SFPLOAD must use INT8 mode (0b0101) to extract
// sign+mag into LREG. SFPABS_MOD1_FLOAT then clears the sign bit — this is the
// "sign-magnitude/float" variant per the SFPABS ISA page (1612186129) and is
// correct for both IEEE floats and sign+mag integers. All other formats go
// through SFPLOAD DEFAULT mode (format picked up from ALU_FORMAT_SPEC_REG).
template <DataFormat fmt = DataFormat::Float32>
inline void _calculate_abs_sfp_rows_()
{
    if constexpr (fmt == DataFormat::Int8 || fmt == DataFormat::UInt8)
    {
        // Dest int8 is sign+magnitude; UInt8 sits in the same SFPLOAD mode bits
        // (0b0101) with bit 15 always 0, so the sign+mag path degenerates to
        // pass-through for unsigned data.
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT8, ADDR_MOD_7, 0, 0);
        TTI_SFPABS(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPABS_MOD1_FLOAT);
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT8, ADDR_MOD_7, 0, 0);
    }
    else if constexpr (fmt == DataFormat::Int32)
    {
        // Dest int32 is sign+magnitude (see Dest storage formats, page 80674824).
        // Smallest negative (-2^31) is saturated to -(2^31-1) at unpack — callers
        // must clamp stimuli to avoid hitting this edge.
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0);
        TTI_SFPABS(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPABS_MOD1_FLOAT);
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0);
    }
    else
    {
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0); // load from dest into lreg[0]

        // Float absolute value (clears sign bit). Use the named constant from
        // sfpi_constants.h — SFPABS_MOD1_INT would do integer 2's-complement abs,
        // which is wrong for float data.
        TTI_SFPABS(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPABS_MOD1_FLOAT);

        // Store result back to destination
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
    }
}

// Implements element-wise absolute value: abs(x)
template <DataFormat fmt = DataFormat::Float32>
inline void _calculate_abs_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_abs_sfp_rows_<fmt>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
