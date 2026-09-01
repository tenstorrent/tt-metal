// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "../field.h"

namespace hal::cfg
{

template <const Field& F, Sec S>
class FieldAssignment;

template <const Field& F, Sec S, std::uint32_t Value>
class ConstantFieldAssignment;

template <const Field& F, Sec S, typename Source>
class GprWrite;

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
