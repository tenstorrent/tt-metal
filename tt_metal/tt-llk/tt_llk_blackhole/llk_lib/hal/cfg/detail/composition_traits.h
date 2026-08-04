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
struct FieldAssignment;

template <const Field& F, std::uint32_t Value, Sec S>
struct ConstantFieldAssignment;

template <RegisterFile File, std::uint32_t Addr, std::uint32_t Mask>
struct ConfigWord;

template <typename Assignment>
struct SingleFieldWord;

} // namespace hal::cfg

namespace hal::cfg::detail
{

template <typename T>
struct is_field_assignment : std::false_type
{
};

template <const Field& F, Sec S>
struct is_field_assignment<FieldAssignment<F, S>> : std::true_type
{
};

template <const Field& F, std::uint32_t Value, Sec S>
struct is_field_assignment<ConstantFieldAssignment<F, Value, S>> : std::true_type
{
};

template <typename T>
struct is_constant_field_assignment : std::false_type
{
};

template <const Field& F, std::uint32_t Value, Sec S>
struct is_constant_field_assignment<ConstantFieldAssignment<F, Value, S>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_field_assignment_v = is_field_assignment<T>::value;

template <typename T>
inline constexpr bool is_constant_field_assignment_v = is_constant_field_assignment<T>::value;

template <typename T>
struct is_config_word : std::false_type
{
};

template <RegisterFile File, std::uint32_t Addr, std::uint32_t Mask>
struct is_config_word<ConfigWord<File, Addr, Mask>> : std::true_type
{
};

template <typename Assignment>
struct is_config_word<SingleFieldWord<Assignment>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_config_word_v = is_config_word<T>::value;

template <typename... Assignments>
struct assignments_disjoint;

template <>
struct assignments_disjoint<> : std::true_type
{
};

template <typename Assignment>
struct assignments_disjoint<Assignment> : std::true_type
{
};

template <typename First, typename... Rest>
struct assignments_disjoint<First, Rest...> : std::bool_constant<(((First::mask & Rest::mask) == 0u) && ...) && assignments_disjoint<Rest...>::value>
{
};

template <typename Assignment>
inline constexpr std::uint32_t encode(const Assignment& assignment)
{
    return (assignment.value << Assignment::shift) & Assignment::mask;
}

} // namespace hal::cfg::detail
