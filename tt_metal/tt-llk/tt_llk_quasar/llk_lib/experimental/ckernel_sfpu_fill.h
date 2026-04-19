// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-02_fill_quasar_7aeae992

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{
// Store fill constant (already loaded in LREG0) to dest for 2 rows
inline void _calculate_fill_sfp_rows_()
{
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
}

// Fill dest with a constant float value
// value: FP32 bit pattern as uint32_t (caller converts float to bits)
inline void _calculate_fill_(const int iterations, const std::uint32_t value)
{
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, (value >> 16));

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_fill_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

// Fill dest with a constant integer value
// STORE_MODE: p_sfpu::sfpmem::INT32 (full 32-bit) or p_sfpu::sfpmem::UINT16 (16-bit unsigned)
// value: integer bit pattern as uint32_t
template <std::uint32_t STORE_MODE>
inline void _calculate_fill_int_(const int iterations, const std::uint32_t value)
{
    // Load fill value into LREG1 (once, before loop)
    if constexpr (STORE_MODE == p_sfpu::sfpmem::INT32)
    {
        TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, (value & 0xFFFF));
        TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, ((value >> 16) & 0xFFFF));
    }
    else if constexpr (STORE_MODE == p_sfpu::sfpmem::UINT16)
    {
        TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, (value & 0xFFFF));
    }

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_7, 0, 0);
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

// Fill dest with a raw 32-bit value (bitcast fill)
// value_bit_mask: raw 32-bit pattern; stored via DEFAULT mode, so the implied
// format (FP16A/FP16B/FP32) determines whether bits are truncated or preserved.
inline void _calculate_fill_bitcast_(const int iterations, const std::uint32_t value_bit_mask)
{
    // Load full 32-bit value into LREG0 (once, before loop)
    TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, (value_bit_mask & 0xFFFF));
    TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, ((value_bit_mask >> 16) & 0xFFFF));

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_fill_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

} // namespace sfpu
} // namespace ckernel
