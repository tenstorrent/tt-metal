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

/**
 * @brief Replace a complete thread-CFG word with an immediate SETC16 instruction.
 *
 * @tparam F: Thread-CFG field identifying the destination word; must start at bit zero.
 * @param section: Section index within F.count.
 * @param value: Complete 16-bit word, already packed into its bit positions.
 * @note Pass section and value that become compile-time constants through inlining.
 */
template <const Field& F>
inline __attribute__((always_inline)) void write_thread_word(const std::uint32_t section, const std::uint32_t value)
{
    LLK_ASSERT(section < F.count, "section index out of range for this register");
    LLK_ASSERT(value <= ThreadCfgWordMask, "prepacked thread CFG value exceeds 16 bits");
    TTI_SETC16(F.addr32(static_cast<Sec>(section)), value & ThreadCfgWordMask);
}

/**
 * @brief Reject the state-reset register as an RMWCIB destination at compile time.
 *
 * @tparam Addr: State-CFG word address, relative to the selected bank.
 * @note Use MMIO or a GPR transfer for state-reset writes, which RMWCIB ignores.
 */
template <std::uint32_t Addr>
inline constexpr void rmwcib_check_address()
{
    static_assert(Addr != STATE_RESET_EN_ADDR32, "RMWCIB writes to the state-reset register are ignored by hardware; use Access::MMIO or from_gpr");
}

/**
 * @brief Update selected state-CFG bits with RMWCIB instructions, one per affected byte.
 *
 * Bytes containing runtime fields use TT_RMWCIB. Constant-only bytes use
 * TTI_RMWCIB. Both paths preserve bits outside Mask and emit bytes from low to high.
 *
 * @tparam Addr: Destination word address, relative to the selected state-CFG bank.
 * @tparam Shamt: Left shift applied to value. Use zero for an already packed group.
 * @tparam Mask: Destination bits to update.
 * @tparam RuntimeMask: Destination bits supplied at runtime. Defaults to Mask.
 * @tparam ConstantData: Already positioned data for constant-only bytes. Defaults to zero.
 * @param value: Data before Shamt, including constant fields in bytes that also contain runtime fields.
 */
template <std::uint32_t Addr, std::uint32_t Shamt, std::uint32_t Mask, std::uint32_t RuntimeMask = Mask, std::uint32_t ConstantData = 0>
inline __attribute__((always_inline)) void rmw_write_word(const std::uint32_t value)
{
    rmwcib_check_address<Addr>();

    const std::uint32_t write_data = value << Shamt;

    if constexpr ((Mask & 0x000000ffu) != 0u)
    {
        if constexpr ((RuntimeMask & 0x000000ffu) != 0u)
        {
            TT_RMWCIB0((Mask >> 0) & 0xffu, (write_data >> 0) & 0xffu, Addr);
        }
        else
        {
            TTI_RMWCIB0((Mask >> 0) & 0xffu, (ConstantData >> 0) & 0xffu, Addr);
        }
    }
    if constexpr ((Mask & 0x0000ff00u) != 0u)
    {
        if constexpr ((RuntimeMask & 0x0000ff00u) != 0u)
        {
            TT_RMWCIB1((Mask >> 8) & 0xffu, (write_data >> 8) & 0xffu, Addr);
        }
        else
        {
            TTI_RMWCIB1((Mask >> 8) & 0xffu, (ConstantData >> 8) & 0xffu, Addr);
        }
    }
    if constexpr ((Mask & 0x00ff0000u) != 0u)
    {
        if constexpr ((RuntimeMask & 0x00ff0000u) != 0u)
        {
            TT_RMWCIB2((Mask >> 16) & 0xffu, (write_data >> 16) & 0xffu, Addr);
        }
        else
        {
            TTI_RMWCIB2((Mask >> 16) & 0xffu, (ConstantData >> 16) & 0xffu, Addr);
        }
    }
    if constexpr ((Mask & 0xff000000u) != 0u)
    {
        if constexpr ((RuntimeMask & 0xff000000u) != 0u)
        {
            TT_RMWCIB3((Mask >> 24) & 0xffu, (write_data >> 24) & 0xffu, Addr);
        }
        else
        {
            TTI_RMWCIB3((Mask >> 24) & 0xffu, (ConstantData >> 24) & 0xffu, Addr);
        }
    }
}

/**
 * @brief Update selected state-CFG bits with immediate RMWCIB instructions.
 *
 * Emit one instruction per byte touched by Mask, from low to high.
 * Bits outside Mask are preserved.
 *
 * @tparam Addr: Destination word address, relative to the selected state-CFG bank.
 * @tparam Mask: Destination bits to update.
 * @tparam Data: Compile-time data already shifted into its destination bit positions.
 */
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

/**
 * @brief Write one CFG word through MMIO, SETC16, or RMWCIB according to access and scope.
 *
 * State-CFG writes preserve bits outside Mask. Thread-CFG writes replace the
 * complete word, clearing bits outside Mask. Single fields remain unshifted until
 * emission to avoid keeping encoded temporaries live across writes.
 *
 * @tparam A: Access::MMIO or Access::TensixCfgUnit. MMIO requires state scope.
 * @tparam Scope: RegisterScope::State or RegisterScope::Thread.
 * @tparam Addr: Destination word address within Scope.
 * @tparam Shamt: Left shift applied to value. Use zero for an already packed group.
 * @tparam Mask: Destination bits supplied by value after shifting.
 * @param value: Data before Shamt.
 * @param cfg: State-CFG bank base for MMIO. Unused for Tensix instructions.
 * @note Serialize competing MMIO updates at the caller. Partial-word MMIO writes use a non-atomic read/modify/write.
 */
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
        TT_SETC16(Addr, ((value << Shamt) & Mask) & ThreadCfgWordMask);
    }
    else
    {
        // One logical word update. Only byte lanes touched by the combined
        // mask produce RMWCIB instructions.
        rmw_write_word<Addr, Shamt, Mask>(value);
    }
}

/**
 * @brief Emit a compile-time CFG write through SETC16 or RMWCIB.
 *
 * @tparam Scope: RegisterScope::Thread selects SETC16. RegisterScope::State selects RMWCIB.
 * @tparam Addr: Destination word address within Scope.
 * @tparam Mask: Bits to update for state CFG. Unused for thread CFG, which replaces the complete word.
 * @tparam Data: Already positioned data. Thread CFG uses its low 16 bits.
 */
template <RegisterScope Scope, std::uint32_t Addr, std::uint32_t Mask, std::uint32_t Data>
inline __attribute__((always_inline)) void write_word()
{
    if constexpr (Scope == RegisterScope::Thread)
    {
        TTI_SETC16(Addr, Data & ThreadCfgWordMask);
    }
    else
    {
        rmw_write_word<Addr, Mask, Data>();
    }
}

/**
 * @brief Copy complete words from an array to consecutive state-CFG locations through MMIO.
 *
 * F selects the starting word. Its mask and bit position do not modify the data.
 * The transfer must fit the bank. If F is wider than one CFG word, the transfer
 * must also stay within the words F occupies.
 *
 * @tparam F: State-CFG field whose addr32(S) selects the first destination word.
 * @tparam S: Section within F.count.
 * @tparam Count: Number of words to copy, no greater than ArrayCount.
 * @tparam ArrayCount: Number of available source words, deduced from values.
 * @param cfg: Destination state-CFG bank base.
 * @param values: Source words. Only the first Count entries are written.
 */
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

/**
 * @brief Transfer one or four consecutive GPR words to state CFG using WRCFG.
 *
 * A compile-time GPR index selects immediate emission. A runtime index selects
 * instruction-buffer emission. Both source and destination must be four-word
 * aligned for a 128-bit transfer.
 *
 * @tparam A: Access::TensixCfgUnit. Blackhole REG2FLOP access is rejected.
 * @tparam F: State-CFG field identifying the first destination word; must start at bit zero.
 * @tparam S: Section within F.count.
 * @tparam GprIndex: Source GPR index, or hal::detail::DynamicGprIndex for a runtime index.
 * @tparam Size: GprTransferSize::Bits32 or GprTransferSize::Bits128.
 * @tparam Completion: WrcfgCompletion::Deferred emits no extra instruction. WrcfgCompletion::Wait appends a NOP.
 * @param transfer: Validated destination operand carrying the source GPR identity.
 * @note Establish the required ordering with GPR producers at the caller before issuing the transfer.
 */
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
        constexpr std::uint32_t wr128b = Size == GprTransferSize::Bits128 ? ckernel::p_cfg::WRCFG_128b : ckernel::p_cfg::WRCFG_32b;

        if constexpr (Size == GprTransferSize::Bits128)
        {
            static_assert((F.addr32(S) & 0x3u) == 0u, "128-bit GPR cfg::write destination must be four-word aligned");
        }
        if constexpr (GprIndex == hal::detail::DynamicGprIndex)
        {
            LLK_ASSERT(transfer.source.index < hal::detail::GprCount, "WRCFG GPR index must be in [0, 63]");
            if constexpr (Size == GprTransferSize::Bits128)
            {
                LLK_ASSERT((transfer.source.index & 0x3u) == 0u, "128-bit WRCFG source GPR must be four-word aligned");
            }
            TT_WRCFG(transfer.source.index, wr128b, F.addr32(S));
        }
        else
        {
            static_assert(GprIndex < hal::detail::GprCount, "WRCFG GPR index must be in [0, 63]");
            if constexpr (Size == GprTransferSize::Bits128)
            {
                static_assert((GprIndex & 0x3u) == 0u, "128-bit WRCFG source GPR must be four-word aligned");
            }
            TTI_WRCFG(GprIndex, wr128b, F.addr32(S));
        }
        if constexpr (Completion == WrcfgCompletion::Wait)
        {
            TTI_NOP;
        }
    }
}

/**
 * @brief Check a runtime field value and accumulate it when its destination group has multiple fields.
 *
 * LLK_ASSERT checks the unshifted value against the field width. Constants are
 * already combined in Plan. Single fields and GPR transfers are handled at emission.
 *
 * @tparam Plan: Compile-time write plan matching the operation sequence.
 * @tparam Index: Position of operation in that sequence.
 * @tparam Operation: Field-assignment or GPR-transfer type, deduced from operation.
 * @param data: Zero-initialized per-group words into which runtime fields are ORed.
 * @param operation: Input operand at Index.
 */
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

/**
 * @brief Emit an operand's destination group only when the operand is the group's first member.
 *
 * Combine precomputed constants with accumulated runtime fields. For state CFG,
 * mixed groups can use immediate RMWCIB instructions for constant-only bytes.
 *
 * @tparam A: Access::MMIO or Access::TensixCfgUnit, validated for the batch.
 * @tparam Plan: Compile-time write plan matching the operation sequence.
 * @tparam Index: Position of operation in that sequence.
 * @tparam Operation: Field-assignment or GPR-transfer type, deduced from operation.
 * @param cfg: State-CFG bank base for MMIO. Unused for Tensix instructions.
 * @param data: Runtime group data prepared by @ref accumulate_write_data.
 * @param operation: Operand at Index, including the value or GPR identity for direct emission.
 */
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
                write_word<A, group.scope, group.addr, 0 /*Shamt*/, group.mask>(group.data, cfg);
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
        else if constexpr (A == Access::TensixCfgUnit && group.scope == RegisterScope::State)
        {
            rmw_write_word<group.addr, 0 /*Shamt*/, group.mask, group.runtime_mask, group.data>(group.data | data[group_index]);
        }
        else
        {
            write_word<A, group.scope, group.addr, 0 /*Shamt*/, group.mask>(group.data | data[group_index], cfg);
        }
    }
}

/**
 * @brief Accumulate runtime fields, then emit groups in first-occurrence order.
 *
 * The unnamed index-sequence argument pairs each operation with its position in Plan.
 * All runtime field checks and accumulation precede the first emitted write.
 *
 * @tparam A: Access::MMIO or Access::TensixCfgUnit, validated for the batch.
 * @tparam Plan: Compile-time write plan matching Operations.
 * @tparam Indices: Consecutive operand positions starting at zero.
 * @tparam Operations: Field-assignment and GPR-transfer types in call order.
 * @param cfg: State-CFG bank base for MMIO. Unused for Tensix instructions.
 * @param operations: Input operands in the order used to construct Plan.
 */
template <Access A, const auto& Plan, std::size_t... Indices, typename... Operations>
inline __attribute__((always_inline)) void write_planned_operations(
    volatile std::uint32_t* tt_reg_ptr cfg, std::index_sequence<Indices...>, const Operations&... operations)
{
    std::array<std::uint32_t, Plan.group_count> data {};
    (accumulate_write_data<Plan, Indices>(data, operations), ...);
    (write_planned_operation<A, Plan, Indices>(cfg, data, operations), ...);
}

/**
 * @brief Validate a CFG write batch and execute its compile-time plan.
 *
 * Reject overlapping destination bits or spans at compile time. Resolve the MMIO
 * bank once for the batch. Field assignments to the same word share a group even
 * when separated by a GPR transfer. Each group emits at its first occurrence.
 *
 * @tparam A: Access::MMIO for state-CFG field assignments, or Access::TensixCfgUnit for field assignments and GPR transfers.
 * @tparam Operations: Operand types produced by @ref set or @ref from_gpr, in call order.
 * @param operations: Field assignments and GPR transfers to execute.
 * @note Split writes into separate calls when hardware programming order must prevent grouping across operands.
 */
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
