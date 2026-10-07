// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "../access_types.h"
// TODO(njokovic) issue #58443: Move instruction-buffer ownership to HAL and remove this dependency.
#include "ckernel.h" // TT_* emission uses ckernel::instrn_buffer.
#include "llk_assert.h"
#include "state_bank.h"
#include "write_operands.h"
#include "write_plan.h"

namespace hal::cfg::detail
{

// Hardware emission: runtime values, MMIO arrays, and constant Tensix instructions.

template <const Field& F>
inline __attribute__((always_inline)) void write_thread_word(const std::uint32_t section, const std::uint32_t value)
{
    LLK_ASSERT(section < F.count, "section index out of range for this register");
    LLK_ASSERT(value <= 0xffffu, "prepacked thread CFG value exceeds 16 bits");
    TTI_SETC16(F.addr32(static_cast<Sec>(section)), value & 0xffffu);
}

template <std::uint32_t Addr>
inline constexpr void rmwcib_check_address()
{
    static_assert(Addr != STATE_RESET_EN_ADDR32, "RMWCIB writes to the state-reset register are ignored by hardware; use Access::MMIO or from_gpr");
}

// Runtime values are shifted into position before emitting the selected byte lanes.
template <std::uint32_t Addr, std::uint32_t Shamt, std::uint32_t Mask>
inline __attribute__((always_inline)) void rmw_write_word(const std::uint32_t value)
{
    rmwcib_check_address<Addr>();

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
    rmwcib_check_address<Addr>();

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
    static_assert(A == Access::MMIO || A == Access::TensixCfgUnit, "composed CFG writes require Access::MMIO or Access::TensixCfgUnit");
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
            // Not atomic against the other RISCs sharing the word.
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

template <const Field& F, Sec S, std::uint32_t Count, std::size_t ArrayCount>
inline __attribute__((always_inline)) void write_array_mmio(volatile std::uint32_t* tt_reg_ptr cfg, const std::array<std::uint32_t, ArrayCount>& values)
{
    static_assert(F.scope == RegisterScope::State, "RISC writes target state CFG");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(Count <= ArrayCount, "CFG word count exceeds source array");

    constexpr std::uint32_t addr = F.addr32(S);
    static_assert(addr < StateCfgWordCount, "CFG array write starts outside the state bank");
    static_assert(Count <= StateCfgWordCount - addr, "CFG array write crosses the end of the state bank");
    static_assert(Count <= anchor_word_limit(F, S), "CFG array write extends past its anchor field");

    for (std::uint32_t i = 0; i < Count; ++i)
    {
        cfg[addr + i] = values[i];
    }
}

// GPR transfers use WRCFG through the CFG unit.

template <Access A, const Field& F, Sec S, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
inline __attribute__((always_inline)) void write_gpr(const GprWrite<F, S, GprIndex, Size, Completion>& transfer)
{
    if constexpr (A == Access::TensixScalarUnit)
    {
        static_assert(A != Access::TensixScalarUnit, "Blackhole does not handle REG2FLOP properly. Use Access::TensixCfgUnit (WRCFG)");
    }
    else
    {
        static_assert(A == Access::TensixCfgUnit, "GPR-backed cfg::write requires Access::TensixCfgUnit");
        if constexpr (Size == GprTransferSize::Bits128)
        {
            static_assert((F.addr32(S) & 0x3u) == 0u, "128-bit GPR cfg::write destination must be four-word aligned");
        }
        if constexpr (GprIndex == hal::detail::DynamicGprIndex)
        {
            LLK_ASSERT(transfer.source.index < 64u, "WRCFG GPR index must be in [0, 63]");
            if constexpr (Size == GprTransferSize::Bits128)
            {
                LLK_ASSERT((transfer.source.index & 0x3u) == 0u, "128-bit WRCFG source GPR must be four-word aligned");
            }
            TT_WRCFG(transfer.source.index, Size == GprTransferSize::Bits128, F.addr32(S));
        }
        else
        {
            static_assert(GprIndex < 64u, "WRCFG GPR index must be in [0, 63]");
            if constexpr (Size == GprTransferSize::Bits128)
            {
                static_assert((GprIndex & 0x3u) == 0u, "128-bit WRCFG source GPR must be four-word aligned");
            }
            TTI_WRCFG(GprIndex, Size == GprTransferSize::Bits128, F.addr32(S));
        }
        if constexpr (Completion == WrcfgCompletion::Wait)
        {
            TTI_NOP;
        }
    }
}

// Validate runtime assignments and accumulate fields that share a group.
// Constants are already combined in the plan; single fields and GPR transfers
// are handled directly at emission.
template <const auto& Plan, std::size_t Index, typename Operation>
inline __attribute__((always_inline)) void accumulate_write_data(std::array<std::uint32_t, Plan.group_count>& data, const Operation& operation)
{
    if constexpr (is_field_assignment_v<Operation> && !is_constant_field_assignment_v<Operation>)
    {
        LLK_ASSERT(operation.value <= (Operation::mask >> Operation::shift), "value exceeds field width");
        if constexpr (Plan.groups[Plan.group_of[Index]].count > 1)
        {
            data[Plan.group_of[Index]] |= encode<Plan.groups[Plan.group_of[Index]].mask>(operation);
        }
    }
}

// Only a group's first operand emits it, preserving first-occurrence order.
template <Access A, const auto& Plan, std::size_t Index, typename Operation>
inline __attribute__((always_inline)) void write_planned_operation(
    volatile std::uint32_t* tt_reg_ptr cfg, const std::array<std::uint32_t, Plan.group_count>& data, const Operation& operation)
{
    constexpr std::size_t group_index = Plan.group_of[Index];
    constexpr auto& group             = Plan.groups[group_index];
    if constexpr (group.first == Index)
    {
        if constexpr (group.all_constant)
        {
            if constexpr (A == Access::TensixCfgUnit)
            {
                write_word<group.scope, group.addr, group.mask, group.data>();
            }
            else
            {
                write_word<A, group.scope, group.addr, 0, group.mask>(group.data, cfg);
            }
        }
        else if constexpr (is_gpr_write_v<Operation>)
        {
            write_gpr<A>(operation);
        }
        else if constexpr (group.count == 1)
        {
            // Preserve the single-field path: shift the value only at emission.
            write_word<A, group.scope, group.addr, Operation::shift, group.mask>(operation.value, cfg);
        }
        else
        {
            write_word<A, group.scope, group.addr, 0, group.mask>(group.data | data[group_index], cfg);
        }
    }
}

template <Access A, const auto& Plan, std::size_t... Indices, typename... Operations>
inline __attribute__((always_inline)) void write_planned_operations(
    volatile std::uint32_t* tt_reg_ptr cfg, std::index_sequence<Indices...>, const Operations&... operations)
{
    std::array<std::uint32_t, Plan.group_count> data {};
    (accumulate_write_data<Plan, Indices>(data, operations), ...);
    (write_planned_operation<A, Plan, Indices>(cfg, data, operations), ...);
}

// Validate one batch, resolve the MMIO bank once, and emit the plan in order.
template <Access A, typename... Operations>
inline __attribute__((always_inline)) void write_operations(const Operations&... operations)
{
    constexpr auto& plan = write_plan_v<Operations...>;
    if constexpr ((is_field_assignment_v<Operations> && ...))
    {
        static_assert(A == Access::MMIO || A == Access::TensixCfgUnit, "field-assignment CFG writes require Access::MMIO or Access::TensixCfgUnit");
        static_assert(plan.disjoint, "overlapping CFG field assignments in one physical word");
    }
    else
    {
        static_assert(A == Access::TensixCfgUnit, "heterogeneous cfg::write supports Access::TensixCfgUnit only");
        static_assert(plan.disjoint, "overlapping field assignments or GPR destination spans in cfg::write");
    }

    volatile std::uint32_t* tt_reg_ptr cfg = nullptr;
    if constexpr (A == Access::MMIO)
    {
        static_assert(((Operations::scope == RegisterScope::State) && ...), "Access::MMIO cannot write thread CFG assignments");
        cfg = state_cfg_bank();
    }
    write_planned_operations<A, plan>(cfg, std::index_sequence_for<Operations...> {}, operations...);
}

} // namespace hal::cfg::detail
