// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace hal
{

/** @brief Select one or both Tensix source register files. */
enum class SourceMask : std::uint8_t
{
    SrcA = 0x1,
    SrcB = 0x2,
    Both = 0x3
};

constexpr SourceMask operator|(const SourceMask lhs, const SourceMask rhs)
{
    return static_cast<SourceMask>(static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs));
}

constexpr SourceMask operator&(const SourceMask lhs, const SourceMask rhs)
{
    return static_cast<SourceMask>(static_cast<std::uint8_t>(lhs) & static_cast<std::uint8_t>(rhs));
}

/** @brief Return the two-bit {B, A} hardware mask of a source selection. */
constexpr std::uint32_t source_bits(const SourceMask sources)
{
    return static_cast<std::uint32_t>(sources);
}

inline constexpr SourceMask SrcA        = SourceMask::SrcA;
inline constexpr SourceMask SrcB        = SourceMask::SrcB;
inline constexpr SourceMask BothSources = SourceMask::Both;

} // namespace hal
