// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-23_fill_quasar_e9608a59
#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi_constants.h"

namespace ckernel
{
namespace sfpu
{

// Broadcast a float constant to all elements of Dest.
// value_bit_mask: IEEE 754 bit-pattern of the float to write.
// Caller converts float->bits via memcpy before calling.
inline void _calculate_fill_(const int iterations, const std::uint32_t value_bit_mask)
{
    // Load 32-bit float bit-pattern into LREG1 before the loop (runtime value, use TT_)
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, value_bit_mask & 0xFFFF);         // lower 16 bits
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, (value_bit_mask >> 16) & 0xFFFF); // upper 16 bits
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0); // write fill constant to dest
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

// Broadcast an integer constant to all elements of Dest with an explicit store mode.
// FMT selects the Dest element type; only Int32 (32-bit) and UInt16 (16-bit) are supported.
template <DataFormat FMT>
inline void _calculate_fill_int_(const int iterations, const std::uint32_t value)
{
    static_assert(FMT == DataFormat::Int32 || FMT == DataFormat::UInt16, "_calculate_fill_int_ supports only DataFormat::Int32 and DataFormat::UInt16");

    constexpr std::uint32_t SFPMEM_MODE = (FMT == DataFormat::UInt16) ? p_sfpu::sfpmem::UINT16 : p_sfpu::sfpmem::INT32;

    if constexpr (FMT == DataFormat::UInt16)
    {
        // 16-bit store: load lower 16 bits only
        TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, value & 0xFFFF);
    }
    else
    {
        // 32-bit store: load full bit-pattern
        TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);         // lower 16 bits
        TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, (value >> 16) & 0xFFFF); // upper 16 bits
    }
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPSTORE(p_sfpu::LREG1, SFPMEM_MODE, ADDR_MOD_7, 0, 0); // write fill constant with explicit format
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

// Broadcast a bit-pattern constant to all elements of Dest.
// Semantically identical to _calculate_fill_; provided for API compatibility.
inline void _calculate_fill_bitcast_(const int iterations, const std::uint32_t value)
{
    _calculate_fill_(iterations, value);
}

} // namespace sfpu
} // namespace ckernel
