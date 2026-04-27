// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-23_fill_quasar_e9608a59
#pragma once

#include <cstdint>
#include <cstring>

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
template <int ITERATIONS>
inline void _calculate_fill_(const float value_bit_mask)
{
    std::uint32_t bits;
    std::memcpy(&bits, &value_bit_mask, sizeof(bits));
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
#pragma GCC unroll 8
    for (std::uint32_t d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

// Broadcast an integer constant to all elements of Dest using an explicit SFPMEM store mode.
// FMT → sfpmem mode:
//   Int32        → sfpmem::INT32  (32-bit sign-magnitude, loads both halves)
//   Int16        → sfpmem::UINT16 (16-bit; INT16 = Quasar hw code 9)
//   Int8 / UInt8 → sfpmem::UINT8  (8-bit)
// Always use SFPLOADI_MOD0_LOWER to place the value at LReg bits [15:0].
// SFPLOADI_MOD0_USHORT (mode 2) shifts the value 10 bits left inside the LReg,
// causing SFPMEM::UINT16 reading [15:0] to produce value<<10 instead of value.
template <DataFormat FMT, int ITERATIONS>
inline void _calculate_fill_int_(const std::uint32_t value)
{
    constexpr std::uint32_t SFPMEM_MODE = (FMT == DataFormat::Int32)   ? p_sfpu::sfpmem::INT32
                                          : (FMT == DataFormat::Int16) ? p_sfpu::sfpmem::UINT16
                                                                       : p_sfpu::sfpmem::UINT8;

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

// Broadcast a bit-pattern constant to all elements of Dest.
// Semantically identical to _calculate_fill_; provided for API compatibility.
// value: Raw bit-pattern to write, not reinterpreted as float.
template <int ITERATIONS>
inline void _calculate_fill_bitcast_(const std::uint32_t value)
{
    float as_float;
    std::memcpy(&as_float, &value, sizeof(as_float));
    _calculate_fill_<ITERATIONS>(as_float);
}

} // namespace sfpu
} // namespace ckernel
