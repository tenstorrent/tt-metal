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
 * @brief The two architectural CFG register scopes.
 *
 * Thread registers are private to the issuing Tensix thread, are 16 bits
 * wide, and are written with SETC16. State registers are shared hardware
 * configuration, are 32 bits wide, and have two banks selected independently
 * by each thread through CFG_STATE_ID.
 */
enum class RegisterScope : std::uint8_t
{
    Thread,
    State
};

// Number of copies of the given register
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

/**
 * @brief A fully self-contained CFG field descriptor.
 *
 * Maps one register field and all its sections to CFG word addresses and bit positions.
 *
 * Section zero:
 *
 *     addr32 = base + word = 64 + 6 = 70
 *     abs0   = addr32 * word_size + shamt0 = 70 * 32 + 4 = 2244
 *
 *     CFG word address 70
 *      31                            8 7       4 3       0
 *     +-------------------------------+---------+---------+
 *     |                               |  field  |         |
 *     +-------------------------------+---------+---------+
 *                                       mask = 0x000000F0
 *                                             |<-- 4 -->|
 *                                             shamt0 = shamt(S0) = 4
 *
 * Section one:
 *
 *     addr32 = (abs0 + sec_bits) / word_size = (2244 + 1536) / 32 = 118
 *     shamt  = (abs0 + sec_bits) % word_size = (2244 + 1536) % 32 = 4
 */
class Field
{
public:
    RegisterScope scope;     // Thread = 16-bit thread CFG, State = 32-bit state CFG
    std::uint32_t word_size; // config word size: 16 (Thread) or 32 (State)
    std::uint32_t base;      // SEC0 register base word
    std::uint32_t word;      // field word within the register (SEC0)
    std::uint32_t shamt0;    // SEC0 bit shift within the word
    std::uint32_t width;     // field width in bits
    std::uint32_t count;     // number of sections
    std::uint32_t sec_bits;  // section stride, in bits

    // Absolute bit offset of the SEC0 field within its CFG scope.
    constexpr std::uint32_t abs0() const
    {
        return (base + word) * word_size + shamt0;
    }

    // CFG word address containing the field in the selected section.
    constexpr std::uint32_t addr32(Sec s) const
    {
        return (abs0() + static_cast<std::uint32_t>(s) * sec_bits) / word_size;
    }

    // Bit offset of the field within its selected section's CFG word.
    constexpr std::uint32_t shamt(Sec s) const
    {
        return (abs0() + static_cast<std::uint32_t>(s) * sec_bits) % word_size;
    }

    // Bit mask selecting the field within its selected section's CFG word.
    constexpr std::uint32_t mask(Sec s) const
    { // valid for width <= 32
        return width >= 32 ? 0xffffffffu : (((1u << width) - 1u) << shamt(s));
    }

    // Number of CFG words occupied by the field in one section.
    constexpr std::uint32_t words() const
    {
        return (shamt0 + width + word_size - 1) / word_size;
    }
};

} // namespace cfg
} // namespace hal
