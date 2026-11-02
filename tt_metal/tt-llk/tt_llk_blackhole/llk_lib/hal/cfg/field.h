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

// Terminology:
//     scope    = logical CFG register block
//     sections = repeated registers within that block
//     bank     = physical/configuration-state copy of the block

/**
 * @brief The two architectural CFG register scopes.
 *
 * Thread registers are private to the issuing Tensix thread, are 16 bits
 * wide, and are written with SETC16: 3 banks, selected by thread id.
 *
 * State registers are shared hardware configuration and are 32 bits wide.
 * The scope covers the two hardware banking schemes, which the HAL addresses
 * identically:
 *
 *     Configuration State (ALU, PACK, UNPACK, THCON) = 2 banks, selected by CFG_STATE_ID
 *     Global                                         = 1 shared bank
 */
enum class RegisterScope : std::uint8_t
{
    Thread,
    State
};

/**
 * @brief Selects one of the repeated registers within a register block.
 */
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
 *                                  mask = 0x000000F0
 *                                               |<-- 4 -->|
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
    std::uint32_t sec_bits;  // distance between consecutive sections, in bits

    // Whether the selected section exists for this field.
    constexpr bool has(Sec s) const
    {
        return static_cast<std::uint32_t>(s) < count;
    }

    // Absolute bit offset of the SEC0 field within its CFG scope.
    constexpr std::uint32_t abs0() const
    {
        return (base + word) * word_size + shamt0;
    }

    // CFG word address containing the field in the selected section.
    constexpr std::uint32_t addr32(Sec s) const
    {
        return (abs0() + section_offset(s)) / word_size;
    }

    // Bit offset of the field within its selected section's CFG word.
    constexpr std::uint32_t shamt(Sec s) const
    {
        return (abs0() + section_offset(s)) % word_size;
    }

    // Bit mask selecting the field within its selected section's CFG word.
    constexpr std::uint32_t mask(Sec s) const
    { // valid for width <= 32
        return width >= 32 ? 0xffffffffu : (((1u << width) - 1u) << shamt(s));
    }

private:
    constexpr std::uint32_t section_offset(Sec s) const
    {
        if (!has(s))
        {
            __builtin_trap();
        }
        return static_cast<std::uint32_t>(s) * sec_bits;
    }
};

/**
 * @brief Extract the selected field from a CFG register word.
 *
 * Uses field F in section S to select the bits from word and return the
 * field value shifted down to bit zero.
 * Example: if the selected field occupies bits 11:8, 0xABCD -> 0x000B.
 */
template <const Field& F, Sec S>
inline constexpr std::uint32_t extract(const std::uint32_t word)
{
    static_assert(F.width <= 32, "field wider than 32b cannot be extracted from a single value");
    static_assert(F.has(S), "section index out of range for this register");
    static_assert(!F.has(S) || F.shamt(S) + F.width <= F.word_size, "field crosses a CFG word boundary");

    return (word & F.mask(S)) >> F.shamt(S);
}

} // namespace cfg
} // namespace hal
