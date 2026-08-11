// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_math_eltwise_sfpu_common.h" // _sfpu_sfpmem_type_<FMT>()
#include "sfpi_constants.h"

namespace ckernel
{
namespace sfpu
{

// Broadcast a 32-bit bit-pattern to all elements of Dest verbatim.
// value: Raw bits written as-is
// detail — the same bits are reassembled and stored via SFPSTORE(sfpmem::DEFAULT).
template <int ITERATIONS>
inline void _calculate_fill_bitcast_(const std::uint32_t value)
{
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, value >> 16);
#pragma GCC unroll 8
    for (std::uint32_t d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

// Broadcast a float constant to all elements of Dest.
template <int ITERATIONS>
inline void _calculate_fill_(const float value)
{
    std::uint32_t bits = __builtin_bit_cast(std::uint32_t, value);
    _calculate_fill_bitcast_<ITERATIONS>(bits);
}

// Broadcast an integer constant to all elements of Dest using an explicit SFPMEM store mode.
// The DataFormat → sfpmem mode is chosen by the canonical _sfpu_sfpmem_type_<FMT>() selector
// (Int32→INT32, Int16→INT16, Int8→INT8, UInt8→UINT8).
// Note: Always use SFPLOADI_MOD0_LOWER to place the value at LReg bits [15:0].
// SFPLOADI_MOD0_USHORT (mode 2) shifts the value 10 bits left inside the LReg,
// causing for example SFPMEM::UINT16 reading [15:0] to produce value<<10 instead of value.
template <DataFormat FMT, int ITERATIONS>
inline void _calculate_fill_int_(const std::uint32_t value)
{
    static_assert(
        FMT == DataFormat::Int32 || FMT == DataFormat::Int16 || FMT == DataFormat::Int8 || FMT == DataFormat::UInt8,
        "_calculate_fill_int_ only supports Int32, Int16, Int8, and UInt8 formats");

    constexpr std::uint32_t SFPMEM_MODE = _sfpu_sfpmem_type_<FMT>();

    if constexpr (FMT == DataFormat::Int32)
    {
        TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, value >> 16);
    }
    else
    {
        TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);
    }
#pragma GCC unroll 8
    for (std::uint32_t d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPSTORE(p_sfpu::LREG1, SFPMEM_MODE, ADDR_MOD_7, 0, 0);
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

} // namespace sfpu
} // namespace ckernel
