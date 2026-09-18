// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../access_types.h"
#include "ckernel.h"
#include "state_bank.h"
#include "write_operands.h"

namespace hal::cfg::detail
{

// Hardware emission: runtime values, MMIO arrays, and constant Tensix instructions.

template <std::uint32_t Addr>
inline constexpr bool rmwcib_is_ignored_by_hardware = (Addr == STATE_RESET_EN_ADDR32);

// Runtime values are shifted into position before emitting the selected byte lanes.
template <std::uint32_t Addr, std::uint32_t Shamt, std::uint32_t Mask>
inline __attribute__((always_inline)) void rmw_write_word(const std::uint32_t value)
{
    static_assert(!rmwcib_is_ignored_by_hardware<Addr>, "RMWCIB writes to the state-reset register are ignored by hardware; use Access::MMIO or from_gpr");

    const std::uint32_t write_data = value << Shamt;

    if constexpr ((Mask & 0x000000ffu) != 0u)
    {
        TT_RMWCIB0((Mask >> 0) & 0xffu, (write_data >> 0) & 0xffu, Addr);
    }
    if constexpr ((Mask & 0x0000ff00u) != 0u)
    {
        TT_RMWCIB1((Mask >> 8) & 0xffu, (write_data >> 8) & 0xffu, Addr);
    }
    if constexpr ((Mask & 0x00ff0000u) != 0u)
    {
        TT_RMWCIB2((Mask >> 16) & 0xffu, (write_data >> 16) & 0xffu, Addr);
    }
    if constexpr ((Mask & 0xff000000u) != 0u)
    {
        TT_RMWCIB3((Mask >> 24) & 0xffu, (write_data >> 24) & 0xffu, Addr);
    }
}

// Compile time known data is already shifted into its destination bit positions.
template <std::uint32_t Addr, std::uint32_t Mask, std::uint32_t Data>
inline __attribute__((always_inline)) void rmw_write_word()
{
    static_assert(!rmwcib_is_ignored_by_hardware<Addr>, "RMWCIB writes to the state-reset register are ignored by hardware; use Access::MMIO or from_gpr");

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

// Single fields keep their value unshifted until emission; composed words use
// Shamt == 0. This avoids extending encoded temporary lifetimes across writes.
template <Access A, RegisterScope Scope, std::uint32_t Addr, std::uint32_t Shamt, std::uint32_t Mask>
inline __attribute__((always_inline)) void write_word(const std::uint32_t value, volatile std::uint32_t* tt_reg_ptr cfg)
{
    static_assert(
        A == Access::MMIO || A == Access::TensixCfgUnit,
        "composed CFG writes require Access::MMIO or Access::TensixCfgUnit; Access::TensixScalarUnit requires a GPR operand");
    if constexpr (A == Access::MMIO)
    {
        static_assert(Scope == RegisterScope::State, "Access::MMIO targets the state CFG; use Access::TensixCfgUnit for thread CFG (SETC16)");
        const std::uint32_t data = (value << Shamt) & Mask;
        if constexpr (Mask == 0xffffffffu)
        {
            cfg[Addr] = data;
        }
        else
        {
            const std::uint32_t old_value = cfg[Addr];
            cfg[Addr]                     = (old_value & ~Mask) | data;
        }
    }
    else if constexpr (Scope == RegisterScope::Thread)
    {
        // SETC16 replaces the complete thread word. Bits absent from Mask are
        // written as zero, matching the existing single-field API.
        TT_SETC16(Addr, ((value << Shamt) & Mask) & 0xffffu);
    }
    else
    {
        // One logical word update. Only byte lanes touched by the combined
        // mask produce RMWCIB instructions.
        rmw_write_word<Addr, Shamt, Mask>(value);
    }
}

template <RegisterScope Scope, std::uint32_t Addr, std::uint32_t Mask, std::uint32_t Data>
inline __attribute__((always_inline)) void write_word()
{
    if constexpr (Scope == RegisterScope::Thread)
    {
        TTI_SETC16(Addr, Data & 0xffffu);
    }
    else
    {
        rmw_write_word<Addr, Mask, Data>();
    }
}

/**
 * @brief Read-modify-write a runtime-masked field of one state-CFG word.
 *
 * Not atomic against the other RISCs sharing the word.
 */
inline void rmw_state_word_mmio(const std::uint32_t addr32, const std::uint32_t shamt, const std::uint32_t mask, const std::uint32_t value)
{
    volatile std::uint32_t* tt_reg_ptr cfg = state_cfg_bank();

    const std::uint32_t old_value = cfg[addr32];
    cfg[addr32]                   = (old_value & ~mask) | ((value << shamt) & mask);
}

template <const Field& F, Sec S, std::uint32_t Count, std::size_t ArrayCount>
inline __attribute__((always_inline)) void write_array_mmio(volatile std::uint32_t* tt_reg_ptr cfg, const std::uint32_t (&values)[ArrayCount])
{
    static_assert(F.scope == RegisterScope::State, "RISC writes target state CFG");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(Count <= ArrayCount, "CFG word count exceeds source array");

    for (std::uint32_t i = 0; i < Count; ++i)
    {
        cfg[F.addr32(S) + i] = values[i];
    }
}

// Field assignments: group by physical word, then emit each group once.

template <typename Tuple, std::size_t Index>
using operand_at_t = std::remove_cv_t<std::remove_reference_t<std::tuple_element_t<Index, Tuple>>>;

// A word is emitted only at its first assignment, preserving first-occurrence order.
template <typename Key, typename Tuple, std::size_t... Indices>
inline constexpr bool word_has_prior_assignment(std::index_sequence<Indices...>)
{
    return (false || ... || assignments_share_word_v<Key, operand_at_t<Tuple, Indices>>);
}

// Combine assignments for Key's word; assignments to other words contribute zero.
template <Access A, typename Key, typename... Assignments>
inline __attribute__((always_inline)) void write_assignment_group(volatile std::uint32_t* tt_reg_ptr cfg, const Key& key, const Assignments&... assignments)
{
    constexpr std::size_t group_size = (0u + ... + assignments_share_word_v<Key, Assignments>);
    constexpr std::uint32_t mask     = (0u | ... | (assignments_share_word_v<Key, Assignments> ? Assignments::mask : 0u));
    constexpr bool group_is_constant = ((!assignments_share_word_v<Key, Assignments> || is_constant_field_assignment_v<Assignments>) && ...);

    if constexpr (A == Access::TensixCfgUnit && group_is_constant)
    {
        constexpr std::uint32_t data = (0u | ... | (assignments_share_word_v<Key, Assignments> ? encode(Assignments {}) : 0u));
        write_word<Key::scope, Key::addr, mask, data>();
    }
    else if constexpr (group_size == 1u)
    {
        write_word<A, Key::scope, Key::addr, Key::shift, mask>(key.value, cfg);
    }
    else
    {
        const std::uint32_t data = (0u | ... | (assignments_share_word_v<Key, Assignments> ? encode(assignments) : 0u));
        write_word<A, Key::scope, Key::addr, 0, mask>(data, cfg);
    }
}

// Visit each assignment; emit its entire word group unless it was already emitted.
template <Access A, std::size_t Index = 0, typename... Assignments>
inline __attribute__((always_inline)) void write_assignment_groups(volatile std::uint32_t* tt_reg_ptr cfg, const Assignments&... assignments)
{
    if constexpr (Index < sizeof...(Assignments))
    {
        using Tuple = std::tuple<Assignments...>;
        using Key   = std::tuple_element_t<Index, Tuple>;
        if constexpr (!word_has_prior_assignment<Key, Tuple>(std::make_index_sequence<Index> {}))
        {
            write_assignment_group<A>(cfg, std::get<Index>(std::tie(assignments...)), assignments...);
        }
        write_assignment_groups<A, Index + 1u>(cfg, assignments...);
    }
}

// Validate and group assignments, resolving the active bank once per MMIO call.
template <Access A, typename... Assignments>
inline __attribute__((always_inline)) void write_assignments(const Assignments&... assignments)
{
    static_assert(A == Access::MMIO || A == Access::TensixCfgUnit, "field-assignment CFG writes require Access::MMIO or Access::TensixCfgUnit");
    static_assert(write_operations_disjoint<Assignments...>::value, "overlapping CFG field assignments in one physical word");
    if constexpr (A == Access::MMIO)
    {
        static_assert(((Assignments::scope == RegisterScope::State) && ...), "Access::MMIO cannot write thread CFG assignments");
    }

    volatile std::uint32_t* tt_reg_ptr cfg = nullptr;
    if constexpr (A == Access::MMIO)
    {
        cfg = state_cfg_bank();
    }
    write_assignment_groups<A>(cfg, assignments...);
}

// GPR transfers: WRCFG through the CFG unit or REG2FLOP through the scalar unit.

template <Access A, const Field& F, Sec S, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
inline __attribute__((always_inline)) void write_gpr(const GprWrite<F, S, GprIndex, Size, Completion>& transfer)
{
    static_assert(
        A == Access::TensixCfgUnit || A == Access::TensixScalarUnit, "GPR-backed cfg::write requires Access::TensixCfgUnit or Access::TensixScalarUnit");
    if constexpr (Size == GprTransferSize::Bits128)
    {
        static_assert((F.addr32(S) & 0x3u) == 0u, "128-bit GPR cfg::write destination must be four-word aligned");
    }

    if constexpr (A == Access::TensixScalarUnit)
    {
        constexpr std::uint32_t address = F.addr32(S);
        static_assert(
            address >= THCON_CFGREG_BASE_ADDR32 && address < GLOBAL_CFGREG_BASE_ADDR32, "Access::TensixScalarUnit supports THCON CFG destinations only");
        if constexpr (Size == GprTransferSize::Bits128)
        {
            static_assert(address + 3u < GLOBAL_CFGREG_BASE_ADDR32, "128-bit REG2FLOP transfer crosses the THCON CFG range");
        }

        constexpr std::uint32_t size_sel   = Size == GprTransferSize::Bits128 ? 0u : 1u;
        constexpr std::uint32_t flop_index = address - THCON_CFGREG_BASE_ADDR32;
        if constexpr (GprIndex == hal::detail::DynamicGprIndex)
        {
            LLK_ASSERT(transfer.source.index < 64u, "REG2FLOP GPR index must be in [0, 63]");
            if constexpr (Size == GprTransferSize::Bits128)
            {
                LLK_ASSERT((transfer.source.index & 0x3u) == 0u, "128-bit REG2FLOP source GPR must be four-word aligned");
            }
            TT_REG2FLOP(size_sel, 0, 0, 0, flop_index, transfer.source.index);
        }
        else
        {
            static_assert(GprIndex < 64u, "REG2FLOP GPR index must be in [0, 63]");
            if constexpr (Size == GprTransferSize::Bits128)
            {
                static_assert((GprIndex & 0x3u) == 0u, "128-bit REG2FLOP source GPR must be four-word aligned");
            }
            TTI_REG2FLOP(size_sel, 0, 0, 0, flop_index, GprIndex);
        }
    }
    else
    {
        if constexpr (GprIndex == hal::detail::DynamicGprIndex)
        {
            TT_WRCFG(transfer.source.index, Size == GprTransferSize::Bits128, F.addr32(S));
        }
        else
        {
            TTI_WRCFG(GprIndex, Size == GprTransferSize::Bits128, F.addr32(S));
        }
        if constexpr (Completion == WrcfgCompletion::Wait)
        {
            TTI_NOP;
        }
    }
}

// Mixed operations: group consecutive field assignments without crossing GprWrite.

// Find the next GprWrite or the end of the operation list.
template <typename Tuple, std::size_t Index>
inline constexpr std::size_t field_assignment_run_end()
{
    if constexpr (Index == std::tuple_size_v<Tuple>)
    {
        return Index;
    }
    else if constexpr (is_field_assignment_v<operand_at_t<Tuple, Index>>)
    {
        return field_assignment_run_end<Tuple, Index + 1u>();
    }
    else
    {
        return Index;
    }
}

template <Access A, std::size_t Start, typename Tuple, std::size_t... Offsets>
inline __attribute__((always_inline)) void write_field_assignment_run(const Tuple& operations, std::index_sequence<Offsets...>)
{
    write_assignments<A>(std::get<Start + Offsets>(operations)...);
}

template <Access A, std::size_t Index, typename Tuple>
inline __attribute__((always_inline)) void write_operation_sequence(const Tuple& operations)
{
    if constexpr (Index < std::tuple_size_v<Tuple>)
    {
        using Operation = operand_at_t<Tuple, Index>;
        if constexpr (is_field_assignment_v<Operation>)
        {
            constexpr std::size_t end = field_assignment_run_end<Tuple, Index>();
            write_field_assignment_run<A, Index>(operations, std::make_index_sequence<end - Index> {});
            write_operation_sequence<A, end>(operations);
        }
        else
        {
            static_assert(is_gpr_write_v<Operation>, "unsupported operation in heterogeneous cfg::write");
            write_gpr<A>(std::get<Index>(operations));
            write_operation_sequence<A, Index + 1u>(operations);
        }
    }
}

template <Access A, typename... Operations>
inline __attribute__((always_inline)) void write_mixed_operations(const Operations&... operations)
{
    static_assert(A == Access::TensixCfgUnit, "heterogeneous cfg::write supports Access::TensixCfgUnit only");
    static_assert((is_write_operation_v<Operations> && ...), "heterogeneous cfg::write accepts only set() and from_gpr() operations");
    static_assert(write_operations_disjoint<Operations...>::value, "overlapping field assignments or GPR destination spans in cfg::write");

    const auto operation_tuple = std::tie(operations...);
    write_operation_sequence<A, 0u>(operation_tuple);
}

} // namespace hal::cfg::detail
