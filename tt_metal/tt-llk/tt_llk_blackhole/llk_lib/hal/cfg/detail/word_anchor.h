// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "../field.h"

namespace hal::cfg::detail
{

/**
 * @brief Detect whether a field group provides a Field descriptor named Raw for its complete register storage.
 *
 * The unnamed template parameter enables detection without rejecting types that lack Raw.
 *
 * @tparam T: Candidate field-group type, without reference or cv qualifiers.
 */
template <typename T, typename = void>
inline constexpr bool has_raw_anchor_v = false;

/**
 * @brief Check the type of an existing Raw member, ignoring its cv qualifiers.
 *
 * @tparam T: Candidate field-group type with a Raw member.
 */
template <typename T>
inline constexpr bool has_raw_anchor_v<T, std::void_t<decltype(T::Raw)>> = std::is_same_v<std::remove_cv_t<decltype(T::Raw)>, Field>;

/**
 * @brief Select the Field descriptor that identifies where a whole-word CFG access starts.
 *
 * A Field is used directly. For a field group, use its Raw descriptor, which
 * describes the complete register storage occupied by the group.
 * See @ref word_anchor for the meaning of an anchor and examples.
 *
 * @tparam Anchor: Field or field group naming the starting register location.
 * @return The supplied Field, or the group's static Raw Field descriptor.
 */
template <const auto& Anchor>
inline constexpr const Field& resolve_word_anchor()
{
    using AnchorType = std::remove_cv_t<std::remove_reference_t<decltype(Anchor)>>;
    if constexpr (std::is_same_v<AnchorType, Field>)
    {
        return Anchor;
    }
    else
    {
        static_assert(has_raw_anchor_v<AnchorType>, "whole-word CFG access requires a Field or a field group with a Raw anchor");
        return AnchorType::Raw;
    }
}

/**
 * @brief Field descriptor that supplies the starting register location for a whole-word CFG access.
 *
 * An anchor is the Field or field group passed to a whole-word read, array
 * write, or GPR write to name where the access starts. The selected Field
 * supplies the register scope and, together with section S, the starting
 * word address through addr32(S). The operation transfers complete words
 * without extracting or updating only the selected field's bits.
 *
 * A Field is used directly. A field group resolves to its Raw Field descriptor:
 * - Thcon[Reg0].TileDescriptor.InDataFormat names a 4-bit field. Using it as
 *   the anchor for read_word reads the complete 32-bit word containing that field.
 * - Thcon[Reg0].TileDescriptor resolves to TileDescriptor.Raw, which describes
 *   all 128 bits of the descriptor. It selects the first word and bounds the
 *   access to the four words occupied by the descriptor.
 *
 * Only a descriptor wider than one CFG word supplies this additional span
 * limit. A descriptor at most one word wide supplies the starting location;
 * the operation still checks the bank boundary. See @ref anchor_word_limit.
 *
 * @tparam Anchor: Field or field group identifying the start of the access.
 * @note For GPR writes, use a descriptor whose field starts at bit zero of its first word.
 */
template <const auto& Anchor>
inline constexpr const Field& word_anchor = resolve_word_anchor<Anchor>();

/**
 * @brief Determine whether the starting descriptor also limits how many CFG words may be accessed.
 *
 * A descriptor wider than one CFG word describes a bounded register range.
 * Count every word it occupies, including partially occupied first and last
 * words. For example, a 128-bit descriptor starting at bit zero occupies four
 * 32-bit words, allowing read offsets 0 through 3 or a transfer of up to four words.
 *
 * A descriptor at most one word wide only identifies the starting location.
 * Return the unlimited sentinel so it adds no span restriction. Callers must
 * still check the selected bank's boundary and their operation-specific limits.
 *
 * @param field: Resolved descriptor from @ref word_anchor.
 * @param section: Valid section within field.count.
 * @return Occupied word count for a descriptor wider than one word, otherwise 0xffffffffu.
 */
inline constexpr std::uint32_t anchor_word_limit(const Field& field, Sec section)
{
    if (field.width <= field.word_size)
    {
        return 0xffffffffu;
    }
    const std::uint64_t bits = std::uint64_t {field.shamt(section)} + field.width;
    return static_cast<std::uint32_t>((bits + field.word_size - 1u) / field.word_size);
}

} // namespace hal::cfg::detail
