// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"

namespace hal::pack
{

/** @brief Select how a data transfer resolves its configuration and address-counter contexts. */
enum class ContextControl : std::uint8_t
{
    Direct                   = 0b00, // contexts come from the descriptor fields
    HardwareCounter          = 0b01, // contexts come from the RTL flops; a tile-closing transfer advances them
    HardwareCounterReset     = 0b10, // reset the RTL-flop context and treat this transfer as a NOP
    HardwareCounterNoAdvance = 0b11  // contexts come from the RTL flops without advancing them
};

/** @brief Select which transfers get zero-padded rows. */
enum class RowPadding : std::uint8_t
{
    None            = 0b000,
    AllTransfers    = 0b001,
    NonConcatenated = 0b010,
    FinalOnly       = 0b011
};

/** @brief Select the granularity a padded row is filled to. */
enum class PaddingAlignment : std::uint8_t
{
    PerRow,
    To16Datums
};

/** @brief Select the Dest row order the read interfaces walk. */
enum class DestAccess : std::uint8_t
{
    RowMajor, // rows 0, 1, 2, 3
    Strided   // rows 0, 8, 16, 24 (untilize)
};

/** @brief Select whether transferred datums retain their values or become zero (ZeroWrite). */
enum class DatumOverride : std::uint8_t
{
    None,
    Zero
};

/** @brief Select whether a transfer opens a new output row or appends to the previous one. */
enum class Concatenation : std::uint8_t
{
    NewRow,
    Append
};

/** @brief Select whether a transfer closes the tile (flush write-aligners, zero-pad to 16B). */
enum class TileBoundary : std::uint8_t
{
    NotLast,
    Last
};

/**
 * @brief Describe one packer data transfer (PACR).
 *
 * Blackhole has one packer with four Dest-read interfaces; @ref interfaces is their 4-bit
 * participation mask, where zero selects all four (under @ref DatumOverride::Zero a zero mask
 * instead activates the documented special mode). The hard span rules live in the packer
 * CFG state, outside this instruction.
 */
struct DataTransfer
{
    std::uint8_t address_modifier      = 0; // 2-bit ADDR_MOD_PACK index
    ContextControl context             = ContextControl::Direct;
    std::uint8_t configuration_context = 0; // [0, 3]; Direct context only
    std::uint8_t counter_context       = 0; // [0, 2]; used when override_thread_id is set
    bool override_thread_id            = false;
    std::uint8_t interfaces            = 0; // 4-bit Dest-read interface mask; 0 = all
    DatumOverride datum_override       = DatumOverride::None;
    DestAccess dest_access             = DestAccess::RowMajor;
    RowPadding padding                 = RowPadding::None;
    PaddingAlignment alignment         = PaddingAlignment::PerRow;
    Concatenation concatenation        = Concatenation::NewRow;
    TileBoundary boundary              = TileBoundary::NotLast;

    /** @brief Encode the transfer. */
    constexpr std::uint32_t get_operation() const;
};

/**
 * @brief Pipeline-ordered register write through the packer (PACR_SETREG, write flavor).
 *
 * The written 32-bit value must have been staged with @ref RegisterWriteValue. The target
 * address forms as {0xFFB, REG_ADDR[11:10], stream_id, REG_ADDR[9:0], 2'b00} from one of the
 * four preprogrammed TDMA address slots.
 *
 * @note No LLK call site exists; the stall-disable chicken bit is never encoded.
 */
struct RegisterWrite
{
    std::uint8_t address_slot = 0; // [0, 3] preprogrammed TDMA addresses
    std::uint8_t stream_id    = 0; // 6-bit stream id / register address bits [17:12]

    /** @brief Encode the write. */
    constexpr std::uint32_t get_operation() const;
};

/** @brief Stage one 16-bit half of the value a @ref RegisterWrite writes (PACR_SETREG, load flavor). */
struct RegisterWriteValue
{
    bool high_half      = false;
    std::uint16_t value = 0;

    /** @brief Encode the staging operation. */
    constexpr std::uint32_t get_operation() const;
};

/**
 * @brief Describe the packer edge window: datum offsets that zero the X-Y plane's edges (SETPKEDGOF).
 *
 * @note The Blackhole and Wormhole LLKs drive edge zeroing through the PCK_EDGE_OFFSET /
 *       TILE_ROW_SET_MAPPING CFG path instead (Quasar and its firmware use this instruction);
 *       wrapped for cross-architecture parity.
 */
struct EdgeWindow
{
    std::uint8_t x_start = 0; // 4-bit offsets
    std::uint8_t x_end   = 0;
    std::uint8_t y_start = 0;
    std::uint8_t y_end   = 0;

    /** @brief Encode the operation. */
    constexpr std::uint32_t get_operation() const;
};

/** @brief Encode a flush of the packer's non-empty write-aligners with no new data (PACR Flush). */
constexpr std::uint32_t flush_write_aligners_operation()
{
    return TT_OP_PACR(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 /* Flush */, 0);
}

/** @brief Flush the packer's non-empty write-aligners with no new data (PACR Flush). */
inline __attribute__((always_inline)) void flush_write_aligners()
{
    INSTRUCTION_WORD(flush_write_aligners_operation());
}

/** @brief Encode a clear of the packer exponent histogram (CLREXPHIST). */
constexpr std::uint32_t clear_exponent_histogram_operation()
{
    return TT_OP_CLREXPHIST;
}

/** @brief Clear the packer exponent histogram (CLREXPHIST). */
inline __attribute__((always_inline)) void clear_exponent_histogram()
{
    INSTRUCTION_WORD(clear_exponent_histogram_operation());
}

namespace detail
{

constexpr bool is_valid(const DataTransfer transfer)
{
    return transfer.address_modifier < 4u && static_cast<std::uint8_t>(transfer.context) < 4u && transfer.configuration_context < 4u &&
           transfer.counter_context < 3u && transfer.interfaces < 16u && static_cast<std::uint8_t>(transfer.padding) < 4u &&
           (transfer.alignment == PaddingAlignment::PerRow || transfer.padding != RowPadding::None);
}

constexpr std::uint32_t encode(const DataTransfer transfer)
{
    const std::uint32_t row_pad_zero = static_cast<std::uint32_t>(transfer.padding) | (transfer.alignment == PaddingAlignment::To16Datums ? 0b100u : 0u);
    return TT_OP_PACR(
        transfer.configuration_context,
        row_pad_zero,
        transfer.dest_access == DestAccess::Strided ? 1u : 0u,
        transfer.address_modifier,
        transfer.counter_context,
        transfer.datum_override == DatumOverride::Zero ? 1u : 0u,
        transfer.interfaces,
        transfer.override_thread_id ? 1u : 0u,
        transfer.concatenation == Concatenation::Append ? 1u : 0u,
        static_cast<std::uint32_t>(transfer.context),
        0u /* Flush: see flush_write_aligners */,
        transfer.boundary == TileBoundary::Last ? 1u : 0u);
}

constexpr bool is_valid(const RegisterWrite write)
{
    return write.address_slot < 4u && write.stream_id < 64u;
}

constexpr std::uint32_t encode(const RegisterWrite write)
{
    return TT_OP_PACR_SETREG(1u /* Push */, 0u, 0u, 0u /* stall normally */, write.address_slot, write.stream_id, 1u /* Flush: required */, 0u);
}

constexpr bool is_valid(const RegisterWriteValue)
{
    return true;
}

constexpr std::uint32_t encode(const RegisterWriteValue value)
{
    // Load flavor: bit 22 selects it, bits 18:3 carry the half's value, bit 2 selects the half.
    const std::uint32_t fields = (1u << 22) | (static_cast<std::uint32_t>(value.value) << 3) | (value.high_half ? 1u << 2 : 0u);
    return TT_OP(0x4a, fields);
}

constexpr bool is_valid(const EdgeWindow window)
{
    return window.x_start < 16u && window.x_end < 16u && window.y_start < 16u && window.y_end < 16u;
}

constexpr std::uint32_t encode(const EdgeWindow window)
{
    return TT_OP_SETPKEDGOF(window.y_end, window.y_start, window.x_end, window.x_start);
}

constexpr void reject_invalid_constant(const bool valid)
{
    if (__builtin_is_constant_evaluated() && !valid)
    {
        __builtin_trap();
    }
}

#ifdef ENABLE_LLK_ASSERT
inline __attribute__((always_inline)) void assert_valid(const DataTransfer transfer)
{
    LLK_ASSERT(transfer.address_modifier < 4u, "PACR address modifier selects one of four ADDR_MOD_PACK sections");
    LLK_ASSERT(transfer.configuration_context < 4u, "PACR configuration context must be in [0, 3]");
    LLK_ASSERT(transfer.counter_context < 3u, "PACR address-counter context must be in [0, 2]");
    LLK_ASSERT(transfer.interfaces < 16u, "PACR Dest-read interface selection is a 4-bit mask");
    LLK_ASSERT(transfer.alignment == PaddingAlignment::PerRow || transfer.padding != RowPadding::None, "16-datum padding alignment requires row padding");
}

inline __attribute__((always_inline)) void assert_valid(const RegisterWrite write)
{
    LLK_ASSERT(write.address_slot < 4u, "PACR_SETREG address slot must be in [0, 3]");
    LLK_ASSERT(write.stream_id < 64u, "PACR_SETREG stream id is 6 bits");
}

inline __attribute__((always_inline)) void assert_valid(const RegisterWriteValue)
{
}

inline __attribute__((always_inline)) void assert_valid(const EdgeWindow window)
{
    LLK_ASSERT(detail::is_valid(window), "SETPKEDGOF edge offsets are four bits each");
}
#endif

} // namespace detail

/** @brief Return whether a packer descriptor is encodable. */
constexpr bool is_valid(const DataTransfer transfer)
{
    return detail::is_valid(transfer);
}

constexpr bool is_valid(const RegisterWrite write)
{
    return detail::is_valid(write);
}

constexpr bool is_valid(const EdgeWindow window)
{
    return detail::is_valid(window);
}

inline constexpr __attribute__((always_inline)) std::uint32_t DataTransfer::get_operation() const
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

inline constexpr __attribute__((always_inline)) std::uint32_t RegisterWrite::get_operation() const
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

inline constexpr __attribute__((always_inline)) std::uint32_t RegisterWriteValue::get_operation() const
{
    return detail::encode(*this);
}

inline constexpr __attribute__((always_inline)) std::uint32_t EdgeWindow::get_operation() const
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

/** @brief Issue a compile-time packer descriptor as one immediate instruction. */
// Compile-time path
template <auto Operation>
inline __attribute__((always_inline)) void run()
{
    static_assert(detail::is_valid(Operation), "invalid packer descriptor");
    constexpr std::uint32_t operation = Operation.get_operation();
    INSTRUCTION_WORD(operation);
}

/** @brief Issue a runtime-selected packer descriptor. */
// Runtime path
template <typename Operation>
inline __attribute__((always_inline)) void run(const Operation operation)
{
    ckernel::instrn_buffer[0] = operation.get_operation();
}

} // namespace hal::pack
