// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

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
template <const Field& F, Sec S, typename Source>
class GprWrite
{
public:
    static_assert(F.scope == RegisterScope::State, "GPR-backed CFG writes require a state-CFG destination");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) == 0, "GPR-backed CFG writes must start at the beginning of a CFG word");

    static constexpr RegisterScope scope = F.scope;
    static constexpr std::uint32_t addr  = F.addr32(S);
    static constexpr std::uint32_t words = Source::size == GprTransferSize::Bits128 ? 4u : 1u;

    Source source;
};

} // namespace hal::cfg

namespace hal::cfg::detail
{

template <typename T>
class is_field_assignment : public std::false_type
{
};

template <const Field& F, Sec S>
class is_field_assignment<FieldAssignment<F, S>> : public std::true_type
{
};

template <const Field& F, Sec S, std::uint32_t Value>
class is_field_assignment<ConstantFieldAssignment<F, S, Value>> : public std::true_type
{
};

template <typename T>
class is_constant_field_assignment : public std::false_type
{
};

template <const Field& F, Sec S, std::uint32_t Value>
class is_constant_field_assignment<ConstantFieldAssignment<F, S, Value>> : public std::true_type
{
};

template <typename T>
inline constexpr bool is_field_assignment_v = is_field_assignment<T>::value;

template <typename T>
inline constexpr bool is_constant_field_assignment_v = is_constant_field_assignment<T>::value;

template <typename T>
class is_gpr_write : public std::false_type
{
};

template <const Field& F, Sec S, typename Source>
class is_gpr_write<GprWrite<F, S, Source>> : public std::true_type
{
};

template <typename T>
inline constexpr bool is_gpr_write_v = is_gpr_write<T>::value;

template <typename T>
inline constexpr bool is_write_operation_v = is_field_assignment_v<T> || is_gpr_write_v<T>;

template <typename Lhs, typename Rhs>
inline constexpr bool assignments_share_word_v = Lhs::scope == Rhs::scope && Lhs::addr == Rhs::addr;

template <typename... Assignments>
class assignment_groups_disjoint;

template <>
class assignment_groups_disjoint<> : public std::true_type
{
};

template <typename Assignment>
class assignment_groups_disjoint<Assignment> : public std::true_type
{
};

template <typename First, typename... Rest>
class assignment_groups_disjoint<First, Rest...>
    : public std::bool_constant<
          ((!assignments_share_word_v<First, Rest> || ((First::mask & Rest::mask) == 0u)) && ...) && assignment_groups_disjoint<Rest...>::value>
{
};

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

template <typename... Operations>
class write_operations_disjoint;

template <>
class write_operations_disjoint<> : public std::true_type
{
};

template <typename Operation>
class write_operations_disjoint<Operation> : public std::true_type
{
};

template <typename First, typename... Rest>
class write_operations_disjoint<First, Rest...>
    : public std::bool_constant<(write_operations_disjoint_pair<First, Rest>() && ...) && write_operations_disjoint<Rest...>::value>
{
};

template <typename Assignment>
inline constexpr std::uint32_t encode(const Assignment& assignment)
{
    return (assignment.value << Assignment::shift) & Assignment::mask;
}

} // namespace hal::cfg::detail
