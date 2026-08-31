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
#include "../composition.h"
#include "ckernel.h"
#include "gpr_operand.h"

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

template <const Field& F, Sec S, std::uint32_t Count, std::size_t ArrayCount>
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

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename Key, typename Assignment>
inline constexpr bool assignment_matches_group_v = assignments_share_word_v<Key, Assignment>;

template <typename Key, typename Assignment>
inline constexpr std::uint32_t encode_group_assignment(const Assignment& assignment)
{
    if constexpr (assignment_matches_group_v<Key, Assignment>)
    {
        return encode(assignment);
    }
    else
    {
        return 0u;
    }
}

template <typename Key, typename Assignment>
inline constexpr std::uint32_t encode_constant_group_assignment()
{
    if constexpr (assignment_matches_group_v<Key, Assignment>)
    {
        static_assert(is_constant_field_assignment_v<Assignment>, "constant CFG group contains a runtime assignment");
        return encode(Assignment {});
    }
    else
    {
        return 0u;
    }
}

template <typename Key, typename Tuple, std::size_t... Indices>
inline constexpr bool assignment_group_seen(std::index_sequence<Indices...>)
{
    return (false || ... || assignment_matches_group_v<Key, remove_cvref_t<std::tuple_element_t<Indices, Tuple>>>);
}

template <Access A, std::size_t KeyIndex, typename Tuple, std::size_t... Indices>
inline __attribute__((always_inline)) void write_assignment_group(
    volatile std::uint32_t* tt_reg_ptr cfg, const Tuple& assignments, std::index_sequence<Indices...>)
{
    using Key = remove_cvref_t<std::tuple_element_t<KeyIndex, Tuple>>;

    constexpr std::size_t group_size = (0u + ... + (assignment_matches_group_v<Key, remove_cvref_t<std::tuple_element_t<Indices, Tuple>>> ? 1u : 0u));
    constexpr std::uint32_t mask =
        (0u | ... |
         (assignment_matches_group_v<Key, remove_cvref_t<std::tuple_element_t<Indices, Tuple>>> ? remove_cvref_t<std::tuple_element_t<Indices, Tuple>>::mask
                                                                                                : 0u));
    constexpr bool group_is_constant =
        (true && ... &&
         (!assignment_matches_group_v<Key, remove_cvref_t<std::tuple_element_t<Indices, Tuple>>> ||
          is_constant_field_assignment_v<remove_cvref_t<std::tuple_element_t<Indices, Tuple>>>));

    if constexpr (A == Access::TensixCfgUnit && group_is_constant)
    {
        constexpr std::uint32_t data = (0u | ... | encode_constant_group_assignment<Key, remove_cvref_t<std::tuple_element_t<Indices, Tuple>>>());
        write_constant_word<Key::file, Key::addr, mask, data>();
    }
    else if constexpr (group_size == 1u)
    {
        const auto& assignment = std::get<KeyIndex>(assignments);
        const SingleFieldWord<Key> single_word {assignment};
        if constexpr (A == Access::MMIO)
        {
            write_word_mmio(single_word, cfg);
        }
        else
        {
            write_word<A>(single_word);
        }
    }
    else
    {
        const std::uint32_t data =
            (0u | ... | encode_group_assignment<Key, remove_cvref_t<std::tuple_element_t<Indices, Tuple>>>(std::get<Indices>(assignments)));
        const ConfigWord<Key::file, Key::addr, mask> grouped_word {data};
        if constexpr (A == Access::MMIO)
        {
            write_word_mmio(grouped_word, cfg);
        }
        else
        {
            write_word<A>(grouped_word);
        }
    }
}

template <Access A, std::size_t Index, typename Tuple>
inline __attribute__((always_inline)) void write_assignment_groups(volatile std::uint32_t* tt_reg_ptr cfg, const Tuple& assignments)
{
    if constexpr (Index < std::tuple_size_v<Tuple>)
    {
        using Key = remove_cvref_t<std::tuple_element_t<Index, Tuple>>;
        if constexpr (!assignment_group_seen<Key, Tuple>(std::make_index_sequence<Index> {}))
        {
            write_assignment_group<A, Index>(cfg, assignments, std::make_index_sequence<std::tuple_size_v<Tuple>> {});
        }
        write_assignment_groups<A, Index + 1u>(cfg, assignments);
    }
}

template <Access A, typename... Assignments>
inline __attribute__((always_inline)) void write_assignments(const Assignments&... assignments)
{
    static_assert(A == Access::MMIO || A == Access::TensixCfgUnit, "field-assignment CFG writes require Access::MMIO or Access::TensixCfgUnit");
    static_assert(assignment_groups_disjoint<Assignments...>::value, "overlapping CFG field assignments in one physical word");
    if constexpr (A == Access::MMIO)
    {
        static_assert(((Assignments::file == RegisterFile::State) && ...), "Access::MMIO cannot write thread CFG assignments");
    }

    const auto assignment_tuple            = std::tie(assignments...);
    volatile std::uint32_t* tt_reg_ptr cfg = nullptr;
    if constexpr (A == Access::MMIO)
    {
        cfg = ckernel::get_cfg_pointer();
    }
    write_assignment_groups<A, 0u>(cfg, assignment_tuple);
}

template <typename... Assignments>
inline __attribute__((always_inline)) void write_assignments_mmio(volatile std::uint32_t* tt_reg_ptr cfg, const Assignments&... assignments)
{
    static_assert(((Assignments::file == RegisterFile::State) && ...), "Access::MMIO cannot write thread CFG assignments");
    static_assert(assignment_groups_disjoint<Assignments...>::value, "overlapping CFG field assignments in one physical word");

    const auto assignment_tuple = std::tie(assignments...);
    write_assignment_groups<Access::MMIO, 0u>(cfg, assignment_tuple);
}

template <Access A, const Field& F, Sec S, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
inline __attribute__((always_inline)) void write_gpr(const GprWrite<F, S, GprOperand<GprIndex, Size, Completion>>& transfer)
{
    static_assert(
        A == Access::TensixCfgUnit || A == Access::TensixScalarUnit, "GPR-backed cfg::write requires Access::TensixCfgUnit or Access::TensixScalarUnit");
    static_assert(F.file == RegisterFile::State, "GPR-backed cfg::write cannot write thread CFG (SETC16) fields");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) == 0, "GPR cfg::write must start at the beginning of a CFG word");
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
        if constexpr (GprIndex == DynamicGprIndex)
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
        if constexpr (GprIndex == DynamicGprIndex)
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

template <typename Tuple, std::size_t Index>
inline constexpr std::size_t assignment_run_end()
{
    if constexpr (Index == std::tuple_size_v<Tuple>)
    {
        return Index;
    }
    else if constexpr (is_field_assignment_v<remove_cvref_t<std::tuple_element_t<Index, Tuple>>>)
    {
        return assignment_run_end<Tuple, Index + 1u>();
    }
    else
    {
        return Index;
    }
}

template <Access A, std::size_t Start, typename Tuple, std::size_t... Offsets>
inline __attribute__((always_inline)) void write_assignment_run(
    volatile std::uint32_t* tt_reg_ptr cfg, const Tuple& operations, std::index_sequence<Offsets...>)
{
    if constexpr (A == Access::MMIO)
    {
        write_assignments_mmio(cfg, std::get<Start + Offsets>(operations)...);
    }
    else
    {
        write_assignments<A>(std::get<Start + Offsets>(operations)...);
    }
}

template <Access A, std::size_t Index, typename Tuple>
inline __attribute__((always_inline)) void write_operation_sequence(volatile std::uint32_t* tt_reg_ptr cfg, const Tuple& operations)
{
    if constexpr (Index < std::tuple_size_v<Tuple>)
    {
        using Operation = remove_cvref_t<std::tuple_element_t<Index, Tuple>>;
        if constexpr (is_field_assignment_v<Operation>)
        {
            constexpr std::size_t end = assignment_run_end<Tuple, Index>();
            write_assignment_run<A, Index>(cfg, operations, std::make_index_sequence<end - Index> {});
            write_operation_sequence<A, end>(cfg, operations);
        }
        else
        {
            static_assert(is_gpr_write_v<Operation>, "unsupported operation in heterogeneous cfg::write");
            write_gpr<A>(std::get<Index>(operations));
            write_operation_sequence<A, Index + 1u>(cfg, operations);
        }
    }
}

template <Access A, typename... Operations>
inline __attribute__((always_inline)) void write_operations(const Operations&... operations)
{
    static_assert(A == Access::TensixCfgUnit, "heterogeneous cfg::write supports Access::TensixCfgUnit only");
    static_assert((is_write_operation_v<Operations> && ...), "heterogeneous cfg::write accepts only set() and from_gpr() operations");
    static_assert(write_operations_disjoint<Operations...>::value, "overlapping field assignments or GPR destination spans in cfg::write");

    const auto operation_tuple             = std::tie(operations...);
    volatile std::uint32_t* tt_reg_ptr cfg = nullptr;
    write_operation_sequence<A, 0u>(cfg, operation_tuple);
}

} // namespace hal::cfg::detail
