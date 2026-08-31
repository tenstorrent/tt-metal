// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// offsets source: tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h
#pragma once

#include <cstdint>

namespace hal
{
namespace cfg
{

/**
 * @brief The two architectural CFG register classes.
 *
 * Thread registers are private to the issuing Tensix thread, are 16 bits
 * wide, and are written with SETC16. State registers are shared hardware
 * configuration, are 32 bits wide, and have two banks selected independently
 * by each thread through CFG_STATE_ID.
 */
enum class RegisterFile : std::uint8_t
{
    Thread,
    State
};

// The section axis, applied via Field::addr32(Sec) / shamt(Sec) / mask(Sec).
enum class Sec : std::uint8_t
{
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7
};

// A fully self-contained field descriptor. It carries its register's file, base
// word, section stride (in bits) and section count, so a field alone locates
// itself — no register type is named alongside it. Storing the stride in bits
// lets bit-packed sections (several sections sharing one word) use the same
// formula as word-strided ones.
struct Field
{
    RegisterFile file;      // Thread = 16-bit thread CFG, State = 32-bit state CFG
    std::uint32_t wbits;    // config word size: 16 (Thread) or 32 (State)
    std::uint32_t base;     // SEC0 register base word
    std::uint32_t word;     // field word within the register (SEC0)
    std::uint32_t shamt0;   // SEC0 bit shift within the word
    std::uint32_t width;    // field width in bits
    std::uint32_t count;    // number of sections
    std::uint32_t sec_bits; // section stride, in bits

    constexpr std::uint32_t abs0() const
    {
        return (base + word) * wbits + shamt0;
    }

    constexpr std::uint32_t addr32(Sec s) const
    {
        return (abs0() + static_cast<std::uint32_t>(s) * sec_bits) / wbits;
    }

    constexpr std::uint32_t shamt(Sec s) const
    {
        return (abs0() + static_cast<std::uint32_t>(s) * sec_bits) % wbits;
    }

    constexpr std::uint32_t mask(Sec s) const
    { // valid for width <= 32
        return width >= 32 ? 0xffffffffu : (((1u << width) - 1u) << shamt(s));
    }

    constexpr std::uint32_t words() const
    {
        return (shamt0 + width + wbits - 1) / wbits;
    }
};

} // namespace cfg
} // namespace hal
