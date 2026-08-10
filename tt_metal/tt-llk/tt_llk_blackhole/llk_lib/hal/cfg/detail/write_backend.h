// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "../access_types.h"
#include "../composition.h"
#include "ckernel.h"

namespace hal::cfg::detail
{

template <std::uint32_t CfgAddr32, std::uint32_t Shamt, std::uint32_t Mask>
inline __attribute__((always_inline)) void cfg_reg_rmw_tensix(const std::uint32_t value)
{
    const std::uint32_t write_data = value << Shamt;

    if constexpr ((Mask & 0x000000ffu) != 0u)
    {
        TT_RMWCIB0((Mask >> 0) & 0xffu, (write_data >> 0) & 0xffu, CfgAddr32);
    }
    if constexpr ((Mask & 0x0000ff00u) != 0u)
    {
        TT_RMWCIB1((Mask >> 8) & 0xffu, (write_data >> 8) & 0xffu, CfgAddr32);
    }
    if constexpr ((Mask & 0x00ff0000u) != 0u)
    {
        TT_RMWCIB2((Mask >> 16) & 0xffu, (write_data >> 16) & 0xffu, CfgAddr32);
    }
    if constexpr ((Mask & 0xff000000u) != 0u)
    {
        TT_RMWCIB3((Mask >> 24) & 0xffu, (write_data >> 24) & 0xffu, CfgAddr32);
    }
}

template <RegisterFile File, std::uint32_t Addr, std::uint32_t Mask>
inline __attribute__((always_inline)) void write_word_mmio(const ConfigWord<File, Addr, Mask>& word, volatile std::uint32_t* tt_reg_ptr cfg)
{
    static_assert(File == RegisterFile::State, "Access::MMIO targets the state CFG; use Access::TensixCfgUnit for thread CFG (SETC16)");
    if constexpr (Mask == 0xffffffffu)
    {
        cfg[Addr] = word.data;
    }
    else
    {
        const std::uint32_t old_value = cfg[Addr];
        cfg[Addr]                     = (old_value & ~Mask) | (word.data & Mask);
    }
}

template <typename Assignment>
inline __attribute__((always_inline)) void write_word_mmio(const SingleFieldWord<Assignment>& word, volatile std::uint32_t* tt_reg_ptr cfg)
{
    static_assert(Assignment::file == RegisterFile::State, "Access::MMIO targets the state CFG; use Access::TensixCfgUnit for thread CFG (SETC16)");
    const std::uint32_t data = encode(word.assignment);
    if constexpr (Assignment::mask == 0xffffffffu)
    {
        cfg[Assignment::addr] = data;
    }
    else
    {
        const std::uint32_t old_value = cfg[Assignment::addr];
        cfg[Assignment::addr]         = (old_value & ~Assignment::mask) | data;
    }
}

template <Access A, RegisterFile File, std::uint32_t Addr, std::uint32_t Mask>
inline __attribute__((always_inline)) void write_word(const ConfigWord<File, Addr, Mask>& word)
{
    static_assert(
        A == Access::MMIO || A == Access::TensixCfgUnit,
        "composed CFG writes require Access::MMIO or Access::TensixCfgUnit; Access::TensixScalarUnit requires a GPR operand");
    if constexpr (A == Access::MMIO)
    {
        write_word_mmio(word, ckernel::get_cfg_pointer());
    }
    else if constexpr (File == RegisterFile::Thread)
    {
        // SETC16 replaces the complete thread word. Bits absent from Mask are
        // written as zero, matching the existing single-field API.
        TT_SETC16(Addr, word.data & 0xffffu);
    }
    else
    {
        // One logical word update. Only byte lanes touched by the combined
        // mask produce RMWCIB instructions.
        cfg_reg_rmw_tensix<Addr, 0, Mask>(word.data);
    }
}

template <Access A, typename Assignment>
inline __attribute__((always_inline)) void write_word(const SingleFieldWord<Assignment>& word)
{
    static_assert(
        A == Access::MMIO || A == Access::TensixCfgUnit,
        "composed CFG writes require Access::MMIO or Access::TensixCfgUnit; Access::TensixScalarUnit requires a GPR operand");
    if constexpr (A == Access::MMIO)
    {
        write_word_mmio(word, ckernel::get_cfg_pointer());
    }
    else if constexpr (Assignment::file == RegisterFile::Thread)
    {
        TT_SETC16(Assignment::addr, encode(word.assignment) & 0xffffu);
    }
    else
    {
        cfg_reg_rmw_tensix<Assignment::addr, Assignment::shift, Assignment::mask>(word.assignment.value);
    }
}

template <typename... Words>
inline __attribute__((always_inline)) void write_words_mmio(volatile std::uint32_t* tt_reg_ptr cfg, const Words&... words)
{
    (write_word_mmio(words, cfg), ...);
}

template <Access A, typename... Words>
inline __attribute__((always_inline)) void write_words(const Words&... words)
{
    if constexpr (A == Access::MMIO)
    {
        write_words_mmio(ckernel::get_cfg_pointer(), words...);
    }
    else
    {
        (write_word<A>(words), ...);
    }
}

template <const Field& F, std::uint32_t Count, Sec S, std::size_t ArrayCount>
inline __attribute__((always_inline)) void write_array_mmio(volatile std::uint32_t* tt_reg_ptr cfg, const std::uint32_t (&values)[ArrayCount])
{
    static_assert(F.file == RegisterFile::State, "RISC writes target state CFG");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(Count <= ArrayCount, "CFG word count exceeds source array");

    for (std::uint32_t i = 0; i < Count; ++i)
    {
        cfg[F.addr32(S) + i] = values[i];
    }
}

template <RegisterFile File, std::uint32_t Addr, std::uint32_t Mask, std::uint32_t Data>
inline __attribute__((always_inline)) void write_constant_word()
{
    if constexpr (File == RegisterFile::Thread)
    {
        TTI_SETC16(Addr, Data & 0xffffu);
    }
    else
    {
        if constexpr ((Mask & 0x000000ffu) != 0u)
        {
            TTI_RMWCIB0((Mask >> 0) & 0xffu, (Data >> 0) & 0xffu, Addr);
        }
        if constexpr ((Mask & 0x0000ff00u) != 0u)
        {
            TTI_RMWCIB1((Mask >> 8) & 0xffu, (Data >> 8) & 0xffu, Addr);
        }
        if constexpr ((Mask & 0x00ff0000u) != 0u)
        {
            TTI_RMWCIB2((Mask >> 16) & 0xffu, (Data >> 16) & 0xffu, Addr);
        }
        if constexpr ((Mask & 0xff000000u) != 0u)
        {
            TTI_RMWCIB3((Mask >> 24) & 0xffu, (Data >> 24) & 0xffu, Addr);
        }
    }
}

template <Access A, typename First, typename... Rest>
inline __attribute__((always_inline)) void write_constant_assignments()
{
    static_assert(A == Access::TensixCfgUnit, "compile-time CFG assignment emission requires Access::TensixCfgUnit");
    static_assert(First::file == RegisterFile::Thread || First::file == RegisterFile::State, "unsupported CFG register file");
    static_assert(((First::file == Rest::file) && ...), "all assignments must use the same register file");
    static_assert(((First::addr == Rest::addr) && ...), "all assignments must resolve to the same physical CFG word");
    static_assert(assignments_disjoint<First, Rest...>::value, "overlapping CFG field assignments");

    constexpr std::uint32_t mask = First::mask | (Rest::mask | ... | 0u);
    constexpr std::uint32_t data = encode(First {}) | (encode(Rest {}) | ... | 0u);

    write_constant_word<First::file, First::addr, mask, data>();
}

} // namespace hal::cfg::detail
