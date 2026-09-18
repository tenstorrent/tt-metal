// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../utils/gpr.h"
#include "access_types.h"
#include "ckernel.h" // RDCFG, SETC16
#include "detail/mmio_read.h"
#include "detail/state_bank.h"
#include "detail/write_backend.h"
#include "detail/write_operands.h"
#include "registers.h"

namespace hal::cfg
{

// Public operand factories and hardware-access entry points.
// F describes a field; S selects its register section. See Field for scope and bank terminology.
// Operand types and implementation helpers live under cfg/detail/.

/*************************************************************************************************
 * Operand factories
 ************************************************************************************************/

/**
 * @brief Construct an assignment of a runtime value to field F in section S.
 */
template <const Field& F, Sec S>
inline constexpr FieldAssignment<F, S> set(const std::uint32_t value)
{
    return {value};
}

/**
 * @brief Construct an assignment of a compile-time value to field F in section S.
 */
template <const Field& F, Sec S, std::uint32_t Value>
inline constexpr ConstantFieldAssignment<F, S, Value> set()
{
    return {};
}

/**
 * @brief Prepare a write from a Tensix GPR to the state-CFG register word identified by F and S.
 *
 * The returned operation can be combined with @ref set assignments in one
 * @ref write call. Field assignments are grouped separately before and after
 * each GPR transfer. The transfer occurs when write() consumes the operation.
 *
 * @tparam F Field descriptor identifying the first destination register word; the field must start at bit zero.
 * @tparam S Register section; must be within F.count.
 * @tparam Size Transfer width: GprTransferSize::Bits32 or GprTransferSize::Bits128; defaults to Bits32.
 * @tparam Completion WRCFG completion policy; defaults to WrcfgCompletion::Wait.
 * @tparam GprIndex GPR index deduced from source.
 * @param source Source GPR operand created with hal::gpr<Index>() or hal::gpr(index).
 */
template <const Field& F, Sec S, GprTransferSize Size = GprTransferSize::Bits32, WrcfgCompletion Completion = WrcfgCompletion::Wait, std::uint32_t GprIndex>
inline constexpr auto from_gpr(const hal::Gpr<GprIndex> source)
{
    return GprWrite<F, S, GprIndex, Size, Completion> {source};
}

/*************************************************************************************************
 * Read interface
 ************************************************************************************************/

/**
 * @brief Read a complete CFG register word through RISC MMIO.
 *
 * The field's mask and bit position are ignored.
 *
 * @tparam A Access path; must be Access::MMIO.
 * @tparam F Field descriptor identifying the source register word.
 * @tparam S Register section; must be within F.count.
 * @tparam WordOffset Word offset relative to the register word containing the field.
 * @tparam Target Thread-CFG bank: Current selects the issuing TRISC; BRISC requires
 *         an explicit T0, T1, or T2. Must be Current for state CFG.
 * @return The 32-bit state-CFG register word or zero-extended 16-bit thread-CFG register word.
 */
template <Access A, const Field& F, Sec S, std::uint32_t WordOffset = 0, ThreadTarget Target = ThreadTarget::Current>
inline __attribute__((always_inline)) std::uint32_t read_word()
{
    static_assert(A == Access::MMIO, "value-returning CFG reads require Access::MMIO");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    if constexpr (F.scope == RegisterScope::Thread)
    {
        static_assert(F.addr32(S) + WordOffset < detail::ThreadCfgWordCount, "thread CFG word offset crosses the selected thread bank");
        return detail::read_thread_word_mmio<Target, F.addr32(S) + WordOffset>() & 0xffffu;
    }
    else
    {
        static_assert(Target == ThreadTarget::Current, "ThreadTarget applies only to thread CFG reads");
        return detail::read_state_word_mmio<F.addr32(S) + WordOffset>();
    }
}

/**
 * @brief Read field F in section S through RISC MMIO.
 *
 * @tparam A Access path; must be Access::MMIO.
 * @tparam F Field descriptor selecting the bits to read.
 * @tparam S Register section; must be within F.count.
 * @tparam Target Thread-CFG bank: Current selects the issuing TRISC; BRISC requires
 *         an explicit T0, T1, or T2. Must be Current for state CFG.
 *
 * @return The field value, shifted down to bit zero.
 */
template <Access A, const Field& F, Sec S, ThreadTarget Target = ThreadTarget::Current>
inline __attribute__((always_inline)) std::uint32_t read()
{
    return extract<F, S>(read_word<A, F, S, 0, Target>());
}

/**
 * @brief Transfer a complete 32-bit state-CFG register word into a Tensix GPR.
 *
 * RDCFG reads from the bank selected by the current thread's CFG_STATE_ID.
 * No field extraction is performed.
 * Use @ref extract to select a field from a register word held as a C++ value.
 *
 * @tparam A Access path; must be Access::TensixCfgUnit.
 * @tparam F Field descriptor identifying the source register word.
 * @tparam S Register section; must be within F.count.
 * @tparam GprIndex Compile-time GPR index deduced from hal::gpr<Index>().
 */
template <Access A, const Field& F, Sec S, std::uint32_t GprIndex>
inline __attribute__((always_inline)) void read(hal::Gpr<GprIndex>)
{
    static_assert(A == Access::TensixCfgUnit, "RDCFG requires Access::TensixCfgUnit");
    static_assert(GprIndex != hal::detail::DynamicGprIndex, "RDCFG requires a compile-time GPR index: use hal::gpr<Index>()");
    static_assert(F.scope == RegisterScope::State, "RDCFG cannot read thread CFG (SETC16) fields");
    static_assert(F.width <= 32, "field wider than 32b cannot be selected through a single CFG word");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) + F.width <= 32, "field crosses a CFG word boundary");

    TTI_RDCFG(GprIndex, F.addr32(S));
}

/*************************************************************************************************
 * Write interface
 ************************************************************************************************/

/**
 * @brief Write a runtime value to field F in the selected register section S.
 *
 * Access::TensixCfgUnit uses TT instructions for this overload.
 *
 * @tparam A Access path: Access::MMIO or Access::TensixCfgUnit.
 * @tparam F Field descriptor selecting the bits to write.
 * @tparam S Register section; must be within F.count.
 * @param value Field value, before shifting to its bit position; must fit the field width.
 *
 * @note MMIO writes support only state CFG. For thread scope, Access::TensixCfgUnit
 *       uses SETC16 to replace the complete 16-bit register word. Other fields are
 *       not preserved; use grouped assignments to write several fields together.
 */
template <Access A, const Field& F, Sec S>
inline __attribute__((always_inline)) void write(const std::uint32_t value)
{
    static_assert(
        A == Access::MMIO || A == Access::TensixCfgUnit,
        "value-backed cfg::write requires Access::MMIO or Access::TensixCfgUnit; Access::TensixScalarUnit requires a GPR operand");
    static_assert(F.width <= 32, "field wider than 32b cannot be written through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    constexpr std::uint32_t cfg_word_addr = F.addr32(S);
    constexpr std::uint32_t max_value     = F.width == 32 ? 0xffffffffu : ((std::uint32_t {1} << F.width) - 1u);
    LLK_ASSERT(value <= max_value, "value exceeds field width");

    if constexpr (A == Access::MMIO)
    {
        static_assert(F.scope == RegisterScope::State, "RISC writes target state CFG; use Access::TensixCfgUnit for thread CFG");
        if constexpr (F.width == 32)
        {
            detail::state_cfg_bank()[cfg_word_addr] = value; // whole 32-bit word
        }
        else
        {
            detail::rmw_state_word_mmio(cfg_word_addr, F.shamt(S), F.mask(S), value); // read-modify-write
        }
    }
    else // Access::TensixCfgUnit
    {
        if constexpr (F.scope == RegisterScope::Thread)
        {
            TT_SETC16(cfg_word_addr, (value << F.shamt(S)) & 0xffffu);
        }
        else
        {
            detail::rmw_write_word<cfg_word_addr, F.shamt(S), F.mask(S)>(value);
        }
    }
}

/**
 * @brief Write a compile-time value to field F in section S using immediate Tensix instructions.
 *
 * @tparam A Access path; must be Access::TensixCfgUnit.
 * @tparam F Field descriptor selecting the bits to write.
 * @tparam S Register section; must be within F.count.
 * @tparam Value Field value, before shifting to its bit position; must fit the field width.
 *
 * @note State CFG uses RMWCIB only for register word bytes covered by the field mask.
 *       Thread CFG uses SETC16 to replace the complete 16-bit register word;
 *       other fields are not preserved.
 */
template <Access A, const Field& F, Sec S, std::uint32_t Value>
inline __attribute__((always_inline)) void write()
{
    static_assert(A == Access::TensixCfgUnit, "compile-time instruction emission requires Access::TensixCfgUnit");
    static_assert(F.width <= 32, "field wider than 32b cannot be written through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(Value <= ((std::uint64_t {1} << F.width) - 1u), "value exceeds field width");

    detail::write_word<F.scope, F.addr32(S), F.mask(S), Value << F.shamt(S)>();
}

/**
 * @brief Group field assignments and emit ordered GPR transfers.
 *
 * Within each consecutive run of field assignments, assignments with the same
 * CFG scope and register word address are combined, even when not adjacent.
 * Register words are written in the order their addresses first appear.
 * With Access::TensixCfgUnit, each register-word group uses TTI instructions
 * when all its assignments are compile-time constants; otherwise it uses TT instructions.
 * Use separate calls when hardware programming order matters.
 * Grouping never crosses a @ref from_gpr operation; its transfer and completion
 * policy are emitted between the surrounding assignment runs.
 *
 * @code
 * write<Access::TensixCfgUnit>(
 *     set<AluFormatSpecReg0::SrcA, Sec::S0>(src_a),
 *     set<AluFormatSpecReg1::SrcB, Sec::S0>(src_b),
 *     set<AluAccCtrl::Fp32_enabled, Sec::S0>(fp32));
 * @endcode
 *
 * @tparam A Access path: Access::MMIO or Access::TensixCfgUnit; from_gpr requires Access::TensixCfgUnit.
 * @tparam First First operation type returned by @ref set or @ref from_gpr.
 * @tparam Rest Remaining operation types.
 * @param first First write operation.
 * @param rest Remaining write operations.
 */
template <
    Access A,
    typename First,
    typename... Rest,
    std::enable_if_t<detail::is_write_operation_v<First> && (detail::is_write_operation_v<Rest> && ...), int> = 0>
inline __attribute__((always_inline)) void write(const First& first, const Rest&... rest)
{
    if constexpr (detail::is_field_assignment_v<First> && (detail::is_field_assignment_v<Rest> && ...))
    {
        detail::write_assignments<A>(first, rest...);
    }
    else
    {
        detail::write_mixed_operations<A>(first, rest...);
    }
}

/**
 * @brief Transfer one or four GPR words to complete state-CFG register words.
 *
 * @tparam A Access path: Access::TensixCfgUnit or Access::TensixScalarUnit.
 * @tparam F Field descriptor identifying the first destination register word; the field must start at bit zero.
 * @tparam S Register section; must be within F.count.
 * @tparam Size Transfer width: GprTransferSize::Bits32 or GprTransferSize::Bits128; defaults to Bits32.
 * @tparam Completion WRCFG completion policy used by Access::TensixCfgUnit; defaults to WrcfgCompletion::Wait.
 * @tparam GprIndex GPR index deduced from source.
 * @param source Source GPR operand created with hal::gpr<Index>() or hal::gpr(index).
 * @note TensixCfgUnit emits WRCFG and its requested completion NOP. TensixScalarUnit
 *       emits REG2FLOP without a completion NOP and accepts only THCON destinations.
 */
template <
    Access A,
    const Field& F,
    Sec S,
    GprTransferSize Size       = GprTransferSize::Bits32,
    WrcfgCompletion Completion = WrcfgCompletion::Wait,
    std::uint32_t GprIndex>
inline __attribute__((always_inline)) void write(const hal::Gpr<GprIndex> source)
{
    detail::write_gpr<A>(from_gpr<F, S, Size, Completion>(source));
}

/**
 * @brief Write Count consecutive state-CFG register words through MMIO.
 *
 * @tparam A Access path; must be Access::MMIO.
 * @tparam F Field descriptor identifying the first destination register word.
 * @tparam S Register section; must be within F.count.
 * @tparam Count Number of register words to write; must not exceed ArrayCount.
 * @tparam ArrayCount Source array length, deduced from values.
 * @param values Complete 32-bit register word values; the field's mask and bit position are ignored.
 */
template <Access A, const Field& F, Sec S, std::uint32_t Count, std::size_t ArrayCount>
inline __attribute__((always_inline)) void write(const std::uint32_t (&values)[ArrayCount])
{
    static_assert(A == Access::MMIO, "array writes require Access::MMIO");
    detail::write_array_mmio<F, S, Count>(detail::state_cfg_bank(), values);
}

} // namespace hal::cfg
