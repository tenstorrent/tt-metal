// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "register_layout.h"
#include "write_operands.h"

namespace hal::cfg::detail
{

struct WriteOperandMeta
{
    RegisterScope scope;
    std::uint32_t addr;
    std::uint32_t mask;
    std::uint32_t words;
    bool is_gpr;
    bool is_constant;
    std::uint32_t data;
};

template <typename Operation>
inline constexpr WriteOperandMeta write_operand_meta()
{
    static_assert(is_write_operation_v<Operation>, "CFG write planning requires set() or from_gpr() operands");

    if constexpr (is_gpr_write_v<Operation>)
    {
        return {Operation::scope, Operation::addr, 0xffffffffu, Operation::words, true, false, 0u};
    }
    else
    {
        constexpr std::uint32_t word_count = Operation::scope == RegisterScope::Thread ? ThreadCfgWordCount : StateCfgWordCount;
        static_assert(Operation::addr < word_count, "CFG write destination lies outside its register scope");

        if constexpr (is_constant_field_assignment_v<Operation>)
        {
            return {Operation::scope, Operation::addr, Operation::mask, 1u, false, true, encode(Operation {})};
        }
        else
        {
            return {Operation::scope, Operation::addr, Operation::mask, 1u, false, false, 0u};
        }
    }
}

struct WriteGroup
{
    RegisterScope scope {};
    std::uint32_t addr = 0;
    std::uint32_t mask = 0;
    std::uint32_t data = 0;
    bool all_constant  = true;
    std::size_t first  = 0;
    std::size_t count  = 0;
};

template <std::size_t Count>
struct WritePlan
{
    // Append each field destination on first use across the entire call.
    // Each GPR transfer gets its own entry in first-occurrence order.
    std::array<WriteGroup, Count> groups {};
    // Direct mapping from each input operand to its output group.
    std::array<std::size_t, Count> group_of {};
    std::size_t group_count = 0;
    bool disjoint           = true;
};

// Each scope has its own fixed word space. Group tracks field operands only
// and is one-based; zero means unseen. Mask also includes GPR writes.
struct WriteWordSlot
{
    std::size_t group  = 0;
    std::uint32_t mask = 0;
};

// O(Count + StateCfgWordCount + ThreadCfgWordCount) constexpr work. GPR spans
// have at most four words. No table or search is needed in the emitted code.
template <std::size_t Count>
inline constexpr WritePlan<Count> build_write_plan(const std::array<WriteOperandMeta, Count>& operands)
{
    WritePlan<Count> plan {};
    std::array<WriteWordSlot, StateCfgWordCount> state {};
    std::array<WriteWordSlot, ThreadCfgWordCount> thread {};

    for (std::size_t i = 0; i < Count; ++i)
    {
        const auto& operand   = operands[i];
        const auto word_count = operand.scope == RegisterScope::Thread ? ThreadCfgWordCount : StateCfgWordCount;
        if (operand.addr >= word_count || operand.words == 0u || operand.words > word_count - operand.addr)
        {
            __builtin_trap();
        }

        // Occupancy covers the entire call, including fields on opposite sides
        // of a GPR transfer. A GPR claims every bit of each destination word.
        for (std::uint32_t k = 0; k < operand.words; ++k)
        {
            auto& slot = operand.scope == RegisterScope::Thread ? thread[operand.addr + k] : state[operand.addr + k];
            if ((slot.mask & operand.mask) != 0u)
            {
                plan.disjoint = false;
            }
            slot.mask |= operand.mask;
        }

        auto& slot        = operand.scope == RegisterScope::Thread ? thread[operand.addr] : state[operand.addr];
        std::size_t group = 0;
        if (operand.is_gpr || slot.group == 0)
        {
            group                    = plan.group_count++;
            plan.groups[group].first = i;
            if (!operand.is_gpr)
            {
                slot.group = group + 1u;
            }
        }
        else
        {
            group = slot.group - 1u;
        }
        plan.group_of[i] = group;

        auto& destination = plan.groups[group];
        destination.scope = operand.scope;
        destination.addr  = operand.addr;
        destination.mask |= operand.mask;
        destination.data |= operand.data;
        destination.all_constant &= operand.is_constant;
        ++destination.count;
    }
    return plan;
}

template <typename... Operations>
inline constexpr auto write_plan_v = build_write_plan(std::array<WriteOperandMeta, sizeof...(Operations)> {{write_operand_meta<Operations>()...}});

} // namespace hal::cfg::detail
