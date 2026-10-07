// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../utils/gpr.h"
#include "access_types.h"
#include "ckernel_ops.h"
#include "detail/mmio_read.h"
#include "detail/state_bank.h"
#include "detail/word_anchor.h"
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
 *
 * @tparam F: Field descriptor selecting the bits to assign.
 * @tparam S: Register section; must be within F.count.
 * @param value: Unshifted field value; must fit the field width.
 */
template <const Field& F, Sec S>
inline constexpr FieldAssignment<F, S> set(const std::uint32_t value)
{
    return {value};
}

/**
 * @brief Construct an assignment of a compile-time value to field F in section S.
 *
 * @tparam F: Field descriptor selecting the bits to assign.
 * @tparam S: Register section; must be within F.count.
 * @tparam Value: Unshifted field value; must fit the field width.
 */
template <const Field& F, Sec S, std::uint32_t Value>
inline constexpr ConstantFieldAssignment<F, S, Value> set()
{
    return {};
}

/**
 * @brief Prepare a write from a Tensix GPR to the state-CFG register word identified by Anchor and S.
 *
 * The returned operation can be combined with @ref set assignments in one
 * @ref write call. Field grouping can span GPR transfers. The transfer occurs
 * when write() consumes the operation.
 *
 * @tparam Anchor: Field, or field group with a Raw anchor, identifying the first destination
 *         register word; it must start at bit zero, and a multi-word anchor must cover the transfer.
 * @tparam S: Register section; must be within the anchor's count.
 * @tparam Size: Transfer width: GprTransferSize::Bits32 or GprTransferSize::Bits128; defaults to Bits32.
 * @tparam Completion: WRCFG completion policy; defaults to WrcfgCompletion::Deferred.
 * @tparam GprIndex: GPR index deduced from source.
 * @param source: Source GPR operand created with hal::gpr<Index>() or hal::gpr(index).
 */
template <
    const auto& Anchor,
    Sec S,
    GprTransferSize Size       = GprTransferSize::Bits32,
    WrcfgCompletion Completion = WrcfgCompletion::Deferred,
    std::uint32_t GprIndex>
inline constexpr auto from_gpr(const hal::Gpr<GprIndex> source)
{
    return GprWrite<detail::word_anchor<Anchor>, S, GprIndex, Size, Completion> {source};
}

/*************************************************************************************************
 * Read interface
 ************************************************************************************************/

/**
 * @brief Read a complete CFG register word through RISC MMIO.
 *
 * The field's mask and bit position are ignored.
 *
 * @tparam A: Access path; must be Access::MMIO.
 * @tparam Anchor: Field, or field group with a Raw anchor, identifying the source register word.
 * @tparam S: Register section; must be within the anchor's count.
 * @tparam WordOffset: Word offset relative to the register word containing the anchor;
 *         must stay within a multi-word anchor.
 * @tparam Target: Thread-CFG bank: Current selects the issuing TRISC; BRISC requires
 *         an explicit T0, T1, or T2. Must be Current for state CFG.
 * @return The 32-bit state-CFG register word or zero-extended 16-bit thread-CFG register word.
 */
template <Access A, const auto& Anchor, Sec S, std::uint32_t WordOffset = 0, ThreadTarget Target = ThreadTarget::Current>
inline __attribute__((always_inline)) std::uint32_t read_word()
{
    constexpr const Field& F = detail::word_anchor<Anchor>;
    static_assert(A == Access::MMIO, "value-returning CFG reads require Access::MMIO");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(WordOffset < detail::anchor_word_limit(F, S), "CFG word offset extends past its anchor field");

    constexpr std::uint64_t addr       = std::uint64_t {F.addr32(S)} + WordOffset;
    constexpr std::uint32_t word_count = F.scope == RegisterScope::Thread ? detail::ThreadCfgWordCount : detail::StateCfgWordCount;
    static_assert(addr < word_count, "CFG word offset crosses the selected bank");

    if constexpr (F.scope == RegisterScope::Thread)
    {
        return detail::read_thread_word_mmio<Target, addr>() & 0xffffu;
    }
    else
    {
        static_assert(Target == ThreadTarget::Current, "ThreadTarget applies only to thread CFG reads");
        return detail::read_state_word_mmio<addr>();
    }
}

/**
 * @brief Read field F in section S through RISC MMIO.
 *
 * @tparam A: Access path; must be Access::MMIO.
 * @tparam F: Field descriptor selecting the bits to read.
 * @tparam S: Register section; must be within F.count.
 * @tparam Target: Thread-CFG bank: Current selects the issuing TRISC; BRISC requires
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
 * @tparam A: Access path; must be Access::TensixCfgUnit.
 * @tparam F: Field descriptor identifying the source register word.
 * @tparam S: Register section; must be within F.count.
 * @tparam GprIndex: Compile-time GPR index deduced from hal::gpr<Index>().
 */
template <Access A, const Field& F, Sec S, std::uint32_t GprIndex>
inline __attribute__((always_inline)) void read(hal::Gpr<GprIndex>)
{
    static_assert(A == Access::TensixCfgUnit, "RDCFG requires Access::TensixCfgUnit");
    static_assert(GprIndex != hal::detail::DynamicGprIndex, "RDCFG requires a compile-time GPR index: use hal::gpr<Index>()");
    static_assert(GprIndex < 64u, "RDCFG GPR index must be in [0, 63]");
    static_assert(F.scope == RegisterScope::State, "RDCFG cannot read thread CFG (SETC16) fields");
    static_assert(F.width <= 32, "field wider than 32b cannot be selected through a single CFG word");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) + F.width <= 32, "field crosses a CFG word boundary");
    static_assert(F.addr32(S) < detail::StateCfgWordCount, "CFG read source lies outside the state bank");

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
 * @tparam A: Access path: Access::MMIO or Access::TensixCfgUnit.
 * @tparam F: Field descriptor selecting the bits to write.
 * @tparam S: Register section; must be within F.count.
 * @param value: Field value, before shifting to its bit position; must fit the field width.
 *
 * @note MMIO writes support only state CFG. For thread scope, Access::TensixCfgUnit
 *       uses SETC16 to replace the complete 16-bit register word. Other fields are
 *       not preserved; use grouped assignments to write several fields together.
 */
template <Access A, const Field& F, Sec S>
inline __attribute__((always_inline)) void write(const std::uint32_t value)
{
    static_assert(A == Access::MMIO || A == Access::TensixCfgUnit, "value-backed cfg::write requires Access::MMIO or Access::TensixCfgUnit");
    static_assert(F.width <= 32, "field wider than 32b cannot be written through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    constexpr std::uint32_t cfg_word_addr = F.addr32(S);
    constexpr std::uint32_t max_value     = F.width == 32 ? 0xffffffffu : ((std::uint32_t {1} << F.width) - 1u);
    LLK_ASSERT(value <= max_value, "value exceeds field width");

    if constexpr (A == Access::MMIO)
    {
        static_assert(F.scope == RegisterScope::State, "RISC writes target state CFG; use Access::TensixCfgUnit for thread CFG");
        detail::write_word<A, F.scope, cfg_word_addr, F.shamt(S), F.mask(S)>(value, detail::state_cfg_bank());
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
 * @brief Write a complete prepacked thread-CFG word using immediate SETC16.
 *
 * Both section and value must become compile-time constants through inlining.
 * The field anchors the register word. Its width does not limit the prepacked value.
 *
 * @tparam A: Access path; must be Access::TensixCfgUnit.
 * @tparam F: Thread-CFG field anchoring the word at bit zero.
 * @param section: Section index; must be smaller than F.count.
 * @param value: Complete 16-bit register word, already packed into its bit positions.
 */
template <Access A, const Field& F>
inline __attribute__((always_inline)) void write(const std::uint32_t section, const std::uint32_t value)
{
    static_assert(A == Access::TensixCfgUnit, "constant-propagated thread CFG writes require Access::TensixCfgUnit");
    static_assert(F.scope == RegisterScope::Thread, "SETC16 targets thread CFG only");
    static_assert(F.shamt(Sec::S0) == 0, "prepacked thread CFG anchor must begin at bit zero");

    detail::write_thread_word<F>(section, value);
}

/**
 * @brief Write a compile-time value to field F in section S using immediate Tensix instructions.
 *
 * @tparam A: Access path; must be Access::TensixCfgUnit.
 * @tparam F: Field descriptor selecting the bits to write.
 * @tparam S: Register section; must be within F.count.
 * @tparam Value: Field value, before shifting to its bit position; must fit the field width.
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
 * @brief Group field assignments across one write batch and emit GPR transfers.
 *
 * Assignments with the same CFG scope and register word address are combined
 * across the entire call, including assignments separated by @ref from_gpr.
 * Field groups and individual GPR transfers are emitted in first-occurrence order.
 * With Access::TensixCfgUnit, each register-word group uses TTI instructions
 * when all its assignments are compile-time constants; otherwise it uses TT instructions.
 * Use separate calls when hardware programming order matters.
 *
 * @note Combining assignments saves instructions only when they share a register
 *       word; assignments to different words gain nothing from sharing one call.
 *
 * @code
 * write<Access::TensixCfgUnit>(
 *     set<AluFormatSpecReg0::SrcA, Sec::S0>(src_a),
 *     set<AluFormatSpecReg1::SrcB, Sec::S0>(src_b),
 *     set<AluAccCtrl::Fp32_enabled, Sec::S0>(fp32));
 * @endcode
 *
 * @tparam A: Access path: Access::MMIO or Access::TensixCfgUnit; from_gpr requires Access::TensixCfgUnit.
 * @tparam First: First operation type returned by @ref set or @ref from_gpr.
 * @tparam Rest: Remaining operation types.
 * @param first: First write operation.
 * @param rest: Remaining write operations.
 */
template <
    Access A,
    typename First,
    typename... Rest,
    std::enable_if_t<detail::is_write_operation_v<First> && (detail::is_write_operation_v<Rest> && ...), int> = 0>
inline __attribute__((always_inline)) void write(const First& first, const Rest&... rest)
{
    detail::write_operations<A>(first, rest...);
}

/**
 * @brief Transfer one or four GPR words to complete state-CFG register words.
 *
 * @tparam A: Access path; must be Access::TensixCfgUnit.
 * @tparam Anchor: Field, or field group with a Raw anchor, identifying the first destination
 *         register word; it must start at bit zero, and a multi-word anchor must cover the transfer.
 * @tparam S: Register section; must be within the anchor's count.
 * @tparam Size: Transfer width: GprTransferSize::Bits32 or GprTransferSize::Bits128; defaults to Bits32.
 * @tparam Completion: WRCFG completion policy; defaults to WrcfgCompletion::Deferred.
 * @tparam GprIndex: GPR index deduced from source.
 * @param source: Source GPR operand created with hal::gpr<Index>() or hal::gpr(index).
 * @note Emits WRCFG and its requested completion NOP. Access::TensixScalarUnit
 *       is unsupported because Blackhole does not handle REG2FLOP properly.
 */
template <
    Access A,
    const auto& Anchor,
    Sec S,
    GprTransferSize Size       = GprTransferSize::Bits32,
    WrcfgCompletion Completion = WrcfgCompletion::Deferred,
    std::uint32_t GprIndex>
inline __attribute__((always_inline)) void write(const hal::Gpr<GprIndex> source)
{
    detail::write_gpr<A>(from_gpr<Anchor, S, Size, Completion>(source));
}

/**
 * @brief Write Count consecutive state-CFG register words from a std::array through MMIO.
 *
 * @code
 * write<Access::MMIO, Thcon[Reg0].TileDescriptor, Sec::S0, TILE_DESC_SIZE>(descriptor_words);
 * @endcode
 *
 * @tparam A: Access path; must be Access::MMIO.
 * @tparam Anchor: Field, or field group with a Raw anchor, identifying the first destination register word.
 * @tparam S: Register section; must be within the anchor's count.
 * @tparam Count: Number of register words to write; must not exceed ArrayCount or a multi-word anchor.
 * @tparam ArrayCount: Source array length, deduced from values.
 * @param values: Complete 32-bit register word values; the field's mask and bit position are ignored.
 */
template <Access A, const auto& Anchor, Sec S, std::uint32_t Count, std::size_t ArrayCount>
inline __attribute__((always_inline)) void write(const std::array<std::uint32_t, ArrayCount>& values)
{
    static_assert(A == Access::MMIO, "array writes require Access::MMIO");
    detail::write_array_mmio<detail::word_anchor<Anchor>, S, Count>(detail::state_cfg_bank(), values);
}

} // namespace hal::cfg
