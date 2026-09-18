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

namespace hal::cfg
{

/**
 * @brief One field assignment, not yet written to hardware.
 *
 * Use @ref set to construct one. Assignments from different generated classes
 * can be combined when their fields occupy the same physical CFG word.
 */
template <const Field& F, Sec S>
class FieldAssignment
{
public:
    static_assert(F.width <= 32, "field wider than 32b cannot be assigned through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    static constexpr RegisterScope scope = F.scope;
    static constexpr std::uint32_t addr  = F.addr32(S);
    static constexpr std::uint32_t shift = F.shamt(S);
    static constexpr std::uint32_t mask  = F.mask(S);

    std::uint32_t value;
};

/**
 * @brief One compile-time field assignment.
 *
 * Unlike @ref FieldAssignment, the value is part of the type. Combining only
 * constant assignments therefore emits immediate TTI_RMWCIB/TTI_SETC16
 * instructions without constructing an opcode at runtime.
 */
template <const Field& F, Sec S, std::uint32_t Value>
class ConstantFieldAssignment
{
public:
    static_assert(F.width <= 32, "field wider than 32b cannot be assigned through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(Value <= ((std::uint64_t {1} << F.width) - 1u), "value exceeds field width");

    static constexpr RegisterScope scope = F.scope;
    static constexpr std::uint32_t addr  = F.addr32(S);
    static constexpr std::uint32_t shift = F.shamt(S);
    static constexpr std::uint32_t mask  = F.mask(S);
    static constexpr std::uint32_t value = Value;
};

/**
 * @brief One destination-bound whole-word GPR transfer.
 *
 * Use @ref from_gpr to construct one. Unlike a field assignment, this operation
 * replaces one or four complete state-CFG words and acts as an ordering barrier
 * between automatically grouped assignment runs.
 */
template <const Field& F, Sec S, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
class GprWrite
{
public:
    static_assert(F.scope == RegisterScope::State, "GPR-backed CFG writes require a state-CFG destination");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) == 0, "GPR-backed CFG writes must start at the beginning of a CFG word");

    static constexpr RegisterScope scope = F.scope;
    static constexpr std::uint32_t addr  = F.addr32(S);
    static constexpr std::uint32_t words = Size == GprTransferSize::Bits128 ? 4u : 1u;

    hal::Gpr<GprIndex> source;
};

} // namespace hal::cfg

namespace hal::cfg::detail
{

// Operand type checks used by write overloads and backend dispatch.

// Reject all types except FieldAssignment and ConstantFieldAssignment.
template <typename T>
inline constexpr bool is_field_assignment_v = false;

// Accept runtime field assignments (FieldAssignment) returned by set().
template <const Field& F, Sec S>
inline constexpr bool is_field_assignment_v<FieldAssignment<F, S>> = true;

// Accept constant field assignments (ConstantFieldAssignment) returned by set().
template <const Field& F, Sec S, std::uint32_t Value>
inline constexpr bool is_field_assignment_v<ConstantFieldAssignment<F, S, Value>> = true;

// Reject all types except ConstantFieldAssignment, including runtime FieldAssignment.
template <typename T>
inline constexpr bool is_constant_field_assignment_v = false;

// Accept ConstantFieldAssignment for immediate instruction emission.
template <const Field& F, Sec S, std::uint32_t Value>
inline constexpr bool is_constant_field_assignment_v<ConstantFieldAssignment<F, S, Value>> = true;

// Reject all types except GprWrite.
template <typename T>
inline constexpr bool is_gpr_write_v = false;

// Accept GPR transfers (GprWrite) returned by from_gpr().
template <const Field& F, Sec S, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
inline constexpr bool is_gpr_write_v<GprWrite<F, S, GprIndex, Size, Completion>> = true;

// Accept FieldAssignment, ConstantFieldAssignment, or GprWrite.
template <typename T>
inline constexpr bool is_write_operation_v = is_field_assignment_v<T> || is_gpr_write_v<T>;

// Destination and overlap checks.

// Assignments share a physical word only when both scope and address match.
template <typename Lhs, typename Rhs>
inline constexpr bool assignments_share_word_v = Lhs::scope == Rhs::scope && Lhs::addr == Rhs::addr;

// Compare FieldAssignment/ConstantFieldAssignment masks or GprWrite word ranges.
template <typename Lhs, typename Rhs>
inline constexpr bool write_operations_disjoint_pair()
{
    if constexpr (is_field_assignment_v<Lhs> && is_field_assignment_v<Rhs>)
    {
        return !assignments_share_word_v<Lhs, Rhs> || ((Lhs::mask & Rhs::mask) == 0u);
    }
    else if constexpr (is_field_assignment_v<Lhs> && is_gpr_write_v<Rhs>)
    {
        return Lhs::scope != Rhs::scope || Lhs::addr < Rhs::addr || Lhs::addr >= Rhs::addr + Rhs::words;
    }
    else if constexpr (is_gpr_write_v<Lhs> && is_field_assignment_v<Rhs>)
    {
        return write_operations_disjoint_pair<Rhs, Lhs>();
    }
    else if constexpr (is_gpr_write_v<Lhs> && is_gpr_write_v<Rhs>)
    {
        return Lhs::scope != Rhs::scope || Lhs::addr + Lhs::words <= Rhs::addr || Rhs::addr + Rhs::words <= Lhs::addr;
    }
    else
    {
        return false;
    }
}

// Require every pair of write operations to have non-overlapping destinations.
template <typename... Operations>
class write_operations_disjoint;

// An empty list has no overlapping operations; this ends the recursion.
template <>
class write_operations_disjoint<> : public std::true_type
{
};

// Check the first operation against the rest, then repeat for the rest.
template <typename First, typename... Rest>
class write_operations_disjoint<First, Rest...>
    : public std::bool_constant<(write_operations_disjoint_pair<First, Rest>() && ...) && write_operations_disjoint<Rest...>::value>
{
};

// Position the value in its field and clear bits outside the field mask.
template <typename Assignment>
inline constexpr std::uint32_t encode(const Assignment& assignment)
{
    return (assignment.value << Assignment::shift) & Assignment::mask;
}

} // namespace hal::cfg::detail
