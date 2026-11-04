// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Write operands, compile-time type checks, and field encoding.

#include <cstdint>
#include <type_traits>

#include "../../utils/gpr.h"
#include "../access_types.h"
#include "../field.h"
#include "register_layout.h"
#include "word_anchor.h"

namespace hal::cfg
{

/**
 * @brief One field assignment, not yet written to hardware.
 *
 * Use @ref set to construct one. Assignments from different generated classes
 * can be combined when their fields occupy the same physical CFG word.
 *
 * @tparam F: Destination field, no wider than 32 bits.
 * @tparam S: Section within F.count.
 */
template <const Field& F, Sec S>
class FieldAssignment
{
public:
    static_assert(F.width <= 32, "field wider than 32b cannot be assigned through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    static constexpr RegisterScope scope = F.scope;     ///< Destination register space.
    static constexpr std::uint32_t addr  = F.addr32(S); ///< Destination word address within scope.
    static constexpr std::uint32_t shift = F.shamt(S);  ///< Left shift that positions the field value in the word.
    static constexpr std::uint32_t mask  = F.mask(S);   ///< Destination bits occupied by the field.

    std::uint32_t value; ///< Unshifted field value, checked when the write consumes this operand.
};

/**
 * @brief One compile-time field assignment.
 *
 * Unlike @ref FieldAssignment, the value is part of the type. With
 * Access::TensixCfgUnit, groups containing only constant assignments emit
 * immediate TTI_RMWCIB/TTI_SETC16 instructions.
 *
 * @tparam F: Destination field, no wider than 32 bits.
 * @tparam S: Section within F.count.
 * @tparam Value: Unshifted field value, checked against the field width at compile time.
 */
template <const Field& F, Sec S, std::uint32_t Value>
class ConstantFieldAssignment
{
public:
    static_assert(F.width <= 32, "field wider than 32b cannot be assigned through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(Value <= ((std::uint64_t {1} << F.width) - 1u), "value exceeds field width");

    static constexpr RegisterScope scope = F.scope;     ///< Destination register space.
    static constexpr std::uint32_t addr  = F.addr32(S); ///< Destination word address within scope.
    static constexpr std::uint32_t shift = F.shamt(S);  ///< Left shift that positions the field value in the word.
    static constexpr std::uint32_t mask  = F.mask(S);   ///< Destination bits occupied by the field.
    static constexpr std::uint32_t value = Value;       ///< Unshifted constant field value.
};

/**
 * @brief One destination-bound whole-word GPR transfer.
 *
 * Use @ref from_gpr to construct one. Unlike a field assignment, this operation
 * replaces one or four complete state-CFG words and gets its own entry in the
 * write plan. Field assignments can be grouped across GPR transfers.
 *
 * @tparam F: State-CFG field identifying the first destination word; must start at bit zero.
 * @tparam S: Section within F.count.
 * @tparam GprIndex: Source GPR index, or hal::detail::DynamicGprIndex for a runtime index.
 * @tparam Size: GprTransferSize::Bits32 or GprTransferSize::Bits128. The span must fit the bank and,
 *         when F is wider than one CFG word, the words that F occupies.
 * @tparam Completion: WrcfgCompletion::Deferred or WrcfgCompletion::Wait, applied when the transfer emits.
 */
template <const Field& F, Sec S, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
class GprWrite
{
public:
    static_assert(F.scope == RegisterScope::State, "GPR-backed CFG writes require a state-CFG destination");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) == 0, "GPR-backed CFG writes must start at the beginning of a CFG word");

    static constexpr RegisterScope scope = F.scope;                                    ///< Destination register space, restricted to state CFG.
    static constexpr std::uint32_t addr  = F.addr32(S);                                ///< First destination word address within the state bank.
    static constexpr std::uint32_t words = Size == GprTransferSize::Bits128 ? 4u : 1u; ///< Number of complete destination words.

    static_assert(addr < detail::StateCfgWordCount, "CFG write destination lies outside its register scope");
    static_assert(words <= detail::StateCfgWordCount - addr, "GPR write crosses the end of its CFG bank");
    static_assert(words <= detail::anchor_word_limit(F, S), "GPR write extends past its anchor field");

    hal::Gpr<GprIndex> source; ///< Source GPR identity. Its index and transfer alignment are checked at emission.
};

} // namespace hal::cfg

namespace hal::cfg::detail
{

// Operand type checks used by write overloads and backend dispatch.

/**
 * @brief Recognize either runtime or constant field-assignment operands.
 *
 * @tparam T: Candidate operand type, without reference or cv qualifiers.
 */
template <typename T>
inline constexpr bool is_field_assignment_v = false;

/**
 * @brief Recognize a runtime field assignment returned by @ref set.
 *
 * @tparam F: Destination field.
 * @tparam S: Selected section.
 */
template <const Field& F, Sec S>
inline constexpr bool is_field_assignment_v<FieldAssignment<F, S>> = true;

/**
 * @brief Recognize a constant field assignment returned by @ref set.
 *
 * @tparam F: Destination field.
 * @tparam S: Selected section.
 * @tparam Value: Unshifted constant field value.
 */
template <const Field& F, Sec S, std::uint32_t Value>
inline constexpr bool is_field_assignment_v<ConstantFieldAssignment<F, S, Value>> = true;

/**
 * @brief Recognize field assignments whose value is encoded in the operand type.
 *
 * @tparam T: Candidate operand type, without reference or cv qualifiers.
 */
template <typename T>
inline constexpr bool is_constant_field_assignment_v = false;

/**
 * @brief Select constant field assignments for compile-time data encoding.
 *
 * @tparam F: Destination field.
 * @tparam S: Selected section.
 * @tparam Value: Unshifted constant field value.
 */
template <const Field& F, Sec S, std::uint32_t Value>
inline constexpr bool is_constant_field_assignment_v<ConstantFieldAssignment<F, S, Value>> = true;

/**
 * @brief Recognize GPR-to-CFG transfer operands.
 *
 * @tparam T: Candidate operand type, without reference or cv qualifiers.
 */
template <typename T>
inline constexpr bool is_gpr_write_v = false;

/**
 * @brief Recognize a GPR transfer returned by @ref from_gpr.
 *
 * @tparam F: State-CFG field identifying the first destination word.
 * @tparam S: Selected section.
 * @tparam GprIndex: Compile-time GPR index or the runtime-index sentinel.
 * @tparam Size: GprTransferSize::Bits32 or GprTransferSize::Bits128.
 * @tparam Completion: WrcfgCompletion::Deferred or WrcfgCompletion::Wait.
 */
template <const Field& F, Sec S, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
inline constexpr bool is_gpr_write_v<GprWrite<F, S, GprIndex, Size, Completion>> = true;

/**
 * @brief Accept field assignments and GPR transfers as operands of a grouped CFG write.
 *
 * @tparam T: Candidate operand type, without reference or cv qualifiers.
 */
template <typename T>
inline constexpr bool is_write_operation_v = is_field_assignment_v<T> || is_gpr_write_v<T>;

/**
 * @brief Position a field value and mask it when it could affect a higher field in the same group.
 *
 * Emission clips the complete word to GroupMask. This helper only needs an
 * additional field mask when the group includes bits above this field.
 *
 * @tparam GroupMask: Union of destination bits for the field's group.
 * @tparam Assignment: Runtime or constant field-assignment type, deduced from assignment.
 * @param assignment: Operand carrying the unshifted field value.
 * @return Positioned value, masked to Assignment::mask when the group contains higher bits.
 */
template <std::uint32_t GroupMask, typename Assignment>
inline constexpr std::uint32_t encode(const Assignment& assignment)
{
    constexpr std::uint32_t through_field = Assignment::mask | (Assignment::mask - 1u);
    const std::uint32_t shifted           = assignment.value << Assignment::shift;
    if constexpr ((GroupMask & ~through_field) != 0u)
    {
        return shifted & Assignment::mask;
    }
    return shifted;
}

} // namespace hal::cfg::detail
