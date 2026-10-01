// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "../field.h"

namespace hal::cfg::detail
{

template <typename T, typename = void>
inline constexpr bool has_raw_anchor_v = false;

template <typename T>
inline constexpr bool has_raw_anchor_v<T, std::void_t<decltype(T::Raw)>> = std::is_same_v<std::remove_cv_t<decltype(T::Raw)>, Field>;

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
 * @brief Field that anchors a whole-word CFG access.
 *
 * A Field anchors itself; a field group such as Thcon[Reg0].TileDescriptor
 * anchors on its Raw field.
 */
template <const auto& Anchor>
inline constexpr const Field& word_anchor = resolve_word_anchor<Anchor>();

/**
 * @brief Maximum word count starting at the field's word in the selected section.
 *
 * A field wider than one CFG word bounds the range to all words it occupies,
 * including a partial first or last word. A narrower field only names its
 * starting word, so the bank alone bounds the range.
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
