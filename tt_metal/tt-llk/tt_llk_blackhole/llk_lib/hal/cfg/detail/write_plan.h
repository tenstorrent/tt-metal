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

/**
 * @brief Describe one write operand using only information available from its type.
 *
 * Runtime field values and GPR identities remain in the original operands.
 * Constant field values are encoded here for use by @ref build_write_plan.
 */
struct WriteOperandMeta
{
    RegisterScope scope; ///< State or thread register space containing the destination.
    std::uint32_t addr;  ///< First destination word address within scope.
    std::uint32_t mask;  ///< Written bits in each destination word. GPR transfers use all bits.
    std::uint32_t words; ///< Destination span in words. Field assignments occupy one word.
    bool is_gpr;         ///< Whether this operand transfers GPR contents instead of a field value.
    bool is_constant;    ///< Whether data contains a compile-time field value.
    std::uint32_t data;  ///< Positioned constant field value, or zero for other operands.
};

/**
 * @brief Extract destination metadata and any constant field data from a write operand type.
 *
 * @tparam Operation: FieldAssignment, ConstantFieldAssignment, or GprWrite specialization.
 * @return Metadata used to group fields and detect overlapping destination bits or spans.
 */
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
        constexpr std::uint32_t word_count = cfg_word_count(Operation::scope);
        static_assert(Operation::addr < word_count, "CFG write destination lies outside its register scope");

        if constexpr (is_constant_field_assignment_v<Operation>)
        {
            return {Operation::scope, Operation::addr, Operation::mask, 1u, false, true, encode<Operation::mask>(Operation {})};
        }
        else
        {
            return {Operation::scope, Operation::addr, Operation::mask, 1u, false, false, 0u};
        }
    }
}

/**
 * @brief Plan one group of field assignments to a word, or one standalone GPR transfer.
 *
 * Field groups combine constant data here and accumulate runtime data at emission.
 * GPR entries always have one operand and do not use the field-data accumulator.
 */
struct WriteGroup
{
    RegisterScope scope {};            ///< Destination register space.
    std::uint32_t addr         = 0;    ///< Destination word address, or first word of a GPR transfer.
    std::uint32_t mask         = 0;    ///< Union of destination bits written by the group's operands.
    std::uint32_t data         = 0;    ///< OR of already positioned constant field values.
    std::uint32_t runtime_mask = 0;    ///< Runtime field bits. GPR entries use the full word mask.
    bool all_constant          = true; ///< True only when every operand is a constant field assignment.
    std::size_t first          = 0;    ///< Input operand index at which this group emits.
    std::size_t count          = 0;    ///< Number of input operands assigned to this group.
};

/**
 * @brief Map an ordered batch of operands to destination groups and record whether they overlap.
 *
 * @tparam Count: Number of input operands and maximum number of output groups.
 */
template <std::size_t Count>
struct WritePlan
{
    std::array<WriteGroup, Count> groups {};    ///< Field groups and standalone GPR transfers in first-occurrence order.
    std::array<std::size_t, Count> group_of {}; ///< Group index for each input operand.
    std::size_t group_count = 0;                ///< Number of populated entries in groups.
    bool disjoint           = true;             ///< False if any operands write overlapping bits in the same register space.
};

/**
 * @brief Track field grouping and occupied bits for one word while constructing a plan.
 *
 * State and thread register spaces have separate slot arrays. GPR transfers
 * occupy bits in every word of their span but do not join field groups.
 */
struct WriteWordSlot
{
    std::size_t group  = 0; ///< Field group index plus one. Zero means no field group has been assigned.
    std::uint32_t mask = 0; ///< Bits occupied by field assignments or GPR transfers.
};

/**
 * @brief Group field destinations across a batch and detect all overlapping writes.
 *
 * Preserve first-occurrence order and give each GPR transfer its own entry.
 * Field grouping spans the entire batch, including operands separated by GPR
 * transfers. Invalid destination spans prevent constant evaluation. Overlapping
 * destinations set WritePlan::disjoint to false for the caller to reject.
 *
 * Construction takes O(Count + StateCfgWordCount + ThreadCfgWordCount) work.
 * When evaluated through @ref write_plan_v, no lookup tables or searches remain
 * in the emitted code.
 *
 * @tparam Count: Number of write operands.
 * @param operands: Metadata in the original call order.
 * @return Destination groups, operand-to-group mapping, and the overlap result.
 */
template <std::size_t Count>
inline constexpr WritePlan<Count> build_write_plan(const std::array<WriteOperandMeta, Count>& operands)
{
    WritePlan<Count> plan {};
    std::array<WriteWordSlot, StateCfgWordCount> state {};
    std::array<WriteWordSlot, ThreadCfgWordCount> thread {};

    for (std::size_t i = 0; i < Count; ++i)
    {
        const auto& operand   = operands[i];
        const auto word_count = cfg_word_count(operand.scope);
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
        if (!operand.is_constant)
        {
            destination.runtime_mask |= operand.mask;
        }
        destination.all_constant &= operand.is_constant;
        ++destination.count;
    }
    return plan;
}

/**
 * @brief Compile-time write plan shared by batches with the same ordered operand types.
 *
 * @tparam Operations: Field-assignment and GPR-transfer types in call order.
 */
template <typename... Operations>
inline constexpr auto write_plan_v = build_write_plan(std::array<WriteOperandMeta, sizeof...(Operations)> {{write_operand_meta<Operations>()...}});

} // namespace hal::cfg::detail
