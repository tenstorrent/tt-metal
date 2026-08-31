// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"

namespace hal::dst
{

/** @brief Select how many Dst rows one zeroing operation touches. */
enum class ZeroScope : std::uint8_t
{
    SingleRow, // current Dst counter plus index (plus configured base offsets)
    Face,      // 16 rows starting at row 16 * index
    Half,      // 512 rows; index bit 0 selects the low or high half
    All        // all 1024 rows; index must be zero
};

/** @brief Select whether rows are marked undefined or re-marked initialized. */
enum class ZeroFlagAction : std::uint8_t
{
    MarkUndefined, // normal zeroing: packer reads 0, FPU reads its identity
    ClearFlags     // re-arm rows as initialized data (budabackend#2730 mitigation path)
};

/** @brief Select 16-bit or 32-bit Dst row addressing. */
enum class DestWidth : std::uint8_t
{
    Bits16,
    Bits32
};

/**
 * @brief Describe one Dst zeroing operation (ZEROACC).
 *
 * ZEROACC does not write datums — it marks the targeted rows through their zero flags:
 * undefined rows read as zero to the packer and as the identity to the FPU. With
 * @ref ZeroFlagAction::ClearFlags the action inverts and previously-zeroed rows are
 * re-marked as initialized data.
 *
 * @note Instruction bits 23:22 clear SrcA/SrcB data valid as an undocumented side effect;
 *       they are always encoded zero here — releases go through @ref hal::src or the
 *       consuming operation.
 */
struct Zero
{
    ZeroScope scope;
    std::uint32_t index       = 0; // row / face / half selector, meaning set by scope
    ZeroFlagAction flags      = ZeroFlagAction::MarkUndefined;
    DestWidth width           = DestWidth::Bits16;
    std::uint8_t address_mode = 0; // applied by hardware only for SingleRow

    /** @brief Encode the operation. */
    constexpr std::uint32_t get_operation() const;
};

namespace detail
{

inline constexpr std::uint32_t SINGLE_ROW_INDEX_LIMIT = 1u << 15; // 15-bit where field
inline constexpr std::uint32_t FACE_INDEX_LIMIT       = 1u << 8;  // where[7:0] selects the face
inline constexpr std::uint32_t HALF_INDEX_LIMIT       = 2u;
inline constexpr std::uint32_t ADDRESS_MODE_COUNT     = 8u;

constexpr bool is_valid(const ZeroScope scope)
{
    return scope == ZeroScope::SingleRow || scope == ZeroScope::Face || scope == ZeroScope::Half || scope == ZeroScope::All;
}

constexpr std::uint32_t index_limit(const ZeroScope scope)
{
    switch (scope)
    {
        case ZeroScope::SingleRow:
            return SINGLE_ROW_INDEX_LIMIT;
        case ZeroScope::Face:
            return FACE_INDEX_LIMIT;
        case ZeroScope::Half:
            return HALF_INDEX_LIMIT;
        default:
            return 1u;
    }
}

constexpr bool is_valid(const Zero operation)
{
    return is_valid(operation.scope) && operation.index < index_limit(operation.scope) && operation.address_mode < ADDRESS_MODE_COUNT &&
           (operation.flags == ZeroFlagAction::MarkUndefined || operation.flags == ZeroFlagAction::ClearFlags) &&
           (operation.width == DestWidth::Bits16 || operation.width == DestWidth::Bits32);
}

constexpr std::uint32_t encode(const Zero operation)
{
    return TT_OP_ZEROACC(
        static_cast<std::uint32_t>(operation.scope),
        operation.width == DestWidth::Bits32 ? 1u : 0u,
        operation.flags == ZeroFlagAction::ClearFlags ? 1u : 0u,
        operation.address_mode,
        operation.index);
}

constexpr void reject_invalid_constant(const bool valid)
{
    if (__builtin_is_constant_evaluated() && !valid)
    {
        __builtin_trap();
    }
}

#ifdef ENABLE_LLK_ASSERT
inline __attribute__((always_inline)) void assert_valid(const Zero operation)
{
    LLK_ASSERT(is_valid(operation.scope), "ZEROACC scope must be SingleRow, Face, Half, or All");
    LLK_ASSERT(operation.index < index_limit(operation.scope), "ZEROACC index exceeds the selected scope's range");
    LLK_ASSERT(operation.address_mode < ADDRESS_MODE_COUNT, "Blackhole ZEROACC address mode must be in [0, 7]");
}
#endif

} // namespace detail

/** @brief Return whether a Dst zeroing descriptor is encodable. */
constexpr bool is_valid(const Zero operation)
{
    return detail::is_valid(operation);
}

inline constexpr __attribute__((always_inline)) std::uint32_t Zero::get_operation() const
{
    detail::reject_invalid_constant(detail::is_valid(*this));
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(*this);
    }
#endif
    return detail::encode(*this);
}

/** @brief Issue a compile-time Dst zeroing operation as one immediate instruction. */
// Compile-time path
template <Zero Operation>
inline __attribute__((always_inline)) void run()
{
    static_assert(detail::is_valid(Operation), "invalid ZEROACC descriptor");
    constexpr std::uint32_t operation = detail::encode(Operation);
    INSTRUCTION_WORD(operation);
}

/** @brief Issue a runtime-selected Dst zeroing operation. */
// Runtime path
inline __attribute__((always_inline)) void run(const Zero operation)
{
    ckernel::instrn_buffer[0] = operation.get_operation();
}

} // namespace hal::dst
