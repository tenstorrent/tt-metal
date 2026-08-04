// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "detail/composition_traits.h"
#include "field.h"

namespace hal::cfg
{

/**
 * @brief Absolute config word of field @p F at section @p S (compile-time).
 */
template <const Field& F, Sec S = Sec::S0>
inline constexpr std::uint32_t word_addr = F.addr32(S);

/**
 * @brief One field assignment, not yet written to hardware.
 *
 * Use @ref set to construct one. Assignments from different generated structs
 * can be combined when their fields occupy the same physical CFG word.
 */
template <const Field& F, Sec S = Sec::S0>
struct FieldAssignment
{
    static_assert(F.width <= 32, "field wider than 32b cannot be assigned through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    static constexpr RegisterFile file   = F.file;
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
template <const Field& F, std::uint32_t Value, Sec S = Sec::S0>
struct ConstantFieldAssignment
{
    static_assert(F.width <= 32, "field wider than 32b cannot be assigned through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(Value <= ((std::uint64_t {1} << F.width) - 1u), "value exceeds field width");

    static constexpr RegisterFile file   = F.file;
    static constexpr std::uint32_t addr  = F.addr32(S);
    static constexpr std::uint32_t shift = F.shamt(S);
    static constexpr std::uint32_t mask  = F.mask(S);
    static constexpr std::uint32_t value = Value;
};

/**
 * @brief Associate a runtime value with a generated CFG field.
 */
template <const Field& F, Sec S = Sec::S0>
inline constexpr FieldAssignment<F, S> set(const std::uint32_t value)
{
    return {value};
}

/**
 * @brief Associate a compile-time value with a generated CFG field.
 */
template <const Field& F, std::uint32_t Value, Sec S = Sec::S0>
inline constexpr ConstantFieldAssignment<F, Value, S> set()
{
    return {};
}

/**
 * @brief A composed physical CFG word.
 *
 * @p Mask is part of the type, allowing the Tensix backend to prune unused
 * RMWCIB byte writes at compile time.
 */
template <RegisterFile File, std::uint32_t Addr, std::uint32_t Mask>
struct ConfigWord
{
    static constexpr RegisterFile file  = File;
    static constexpr std::uint32_t addr = Addr;
    static constexpr std::uint32_t mask = Mask;

    std::uint32_t data;
};

/**
 * @brief A lazily encoded word containing exactly one field assignment.
 *
 * Keeping the unshifted value until emission lets a multi-word write compile
 * like the equivalent sequence of typed single-field writes. In particular,
 * it avoids extending encoded temporary lifetimes across later word updates.
 */
template <typename Assignment>
struct SingleFieldWord
{
    static constexpr RegisterFile file  = Assignment::file;
    static constexpr std::uint32_t addr = Assignment::addr;
    static constexpr std::uint32_t mask = Assignment::mask;

    Assignment assignment;
};

/**
 * @brief Wrap one assignment as a lazily encoded physical CFG word.
 */
template <typename Assignment, std::enable_if_t<detail::is_field_assignment_v<Assignment>, int> = 0>
inline constexpr auto word(const Assignment& assignment)
{
    return SingleFieldWord<Assignment> {assignment};
}

/**
 * @brief Combine assignments occupying the same physical CFG word.
 *
 * Fields may come from different generated structs. Their register file and
 * resolved word address must agree, and their masks must not overlap.
 */
template <typename First, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
inline constexpr auto word(const First& first, const Rest&... rest)
{
    static_assert(
        detail::is_field_assignment_v<First> && (detail::is_field_assignment_v<Rest> && ...), "cfg::word() accepts only values returned by cfg::set()");
    static_assert(((First::file == Rest::file) && ...), "all assignments in cfg::word() must use the same register file");
    static_assert(((First::addr == Rest::addr) && ...), "all assignments in cfg::word() must resolve to the same physical CFG word");
    static_assert(detail::assignments_disjoint<First, Rest...>::value, "overlapping CFG field assignments in cfg::word()");

    constexpr std::uint32_t combined_mask = First::mask | (Rest::mask | ... | 0u);
    return ConfigWord<First::file, First::addr, combined_mask> {detail::encode(first) | (detail::encode(rest) | ... | 0u)};
}

/**
 * @brief Wrap a prepacked complete CFG word using field @p F as its address.
 *
 * The field must begin at bit zero of the selected word. No field mask is
 * applied: writing the result replaces all 32 bits.
 */
template <const Field& F, Sec S = Sec::S0>
inline constexpr auto word(const std::uint32_t value)
{
    static_assert(F.file == RegisterFile::State, "prepacked CFG words target the state CFG");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) == 0, "prepacked CFG word anchor must begin at bit zero");

    return ConfigWord<F.file, F.addr32(S), 0xffffffffu> {value};
}

/**
 * @brief Wrap a prepacked complete CFG word at an offset from field @p F.
 */
template <const Field& F, std::uint32_t WordOffset, Sec S = Sec::S0>
inline constexpr auto word_at(const std::uint32_t value)
{
    static_assert(F.file == RegisterFile::State, "prepacked CFG words target the state CFG");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    return ConfigWord<F.file, F.addr32(S) + WordOffset, 0xffffffffu> {value};
}

} // namespace hal::cfg
