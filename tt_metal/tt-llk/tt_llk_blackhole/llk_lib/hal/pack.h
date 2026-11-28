// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "utils/helpers.h"
#include "utils/operation.h"
#include "utils/validate.h"

namespace hal::pack
{

/**
 * @brief Select the Dst rows a data transfer reads, relative to its start row.
 *
 * Each enabled read interface packs one Dst row; only the masks the Blackhole ISA supports are listed.
 */
enum class ReadInterfaces : std::uint8_t
{
    All      = 0b0000, ///< Rows 0 to 3; with DatumOverride::Zero, the X-range zero mode instead.
    Row0     = 0b0001, ///< Row 0.
    Row0To1  = 0b0011, ///< Rows 0 and 1.
    Row0To2  = 0b0111, ///< Rows 0 to 2.
    Row0And2 = 0b0101, ///< Rows 0 and 2.
    Row1And3 = 0b1010  ///< Rows 1 and 3.
};

/** @brief Select whether transferred datums keep their Dst values or are packed as zeros read from nothing. */
enum class DatumOverride : bool
{
    None = false,
    Zero = true
};

/** @brief Select whether the read interfaces read adjacent Dst rows or rows 8 apart. */
enum class DestAccess : bool
{
    RowMajor = false,
    Strided  = true ///< Rows 8 apart (16 with remapped addresses), for untilize.
};

/** @brief Select whether the next transfer continues the current output row or starts a new one. */
enum class Concatenation : bool
{
    NewRow = false,
    Append = true
};

/** @brief Select whether a transfer closes the tile. */
enum class TileBoundary : bool
{
    NotLast = false,
    Last    = true ///< Flush the output buffers to L1; the next transfer starts at fresh output addresses.
};

/**
 * @brief Describe the options of one PACR data transfer from Dst to L1.
 *
 * Formats, L1 address, edge masking and the Dst read position remain in packer configuration and
 * address-counter state.
 */
class Transfer
{
public:
    std::uint8_t address_modifier = 0;                     ///< ADDR_MOD_PACK section applied after the transfer, in [0, 3].
    ReadInterfaces interfaces     = ReadInterfaces::All;   ///< Dst rows read, relative to the start row.
    DatumOverride datum_override  = DatumOverride::None;   ///< Whether datums keep their values or become zero.
    DestAccess dest_access        = DestAccess::RowMajor;  ///< Adjacent or strided Dst rows.
    Concatenation concatenation   = Concatenation::NewRow; ///< Whether the next transfer continues this output row.
    TileBoundary boundary         = TileBoundary::NotLast; ///< Whether this transfer closes the tile.
};

namespace detail
{
constexpr bool is_valid(const ReadInterfaces interfaces)
{
    return interfaces == ReadInterfaces::All || interfaces == ReadInterfaces::Row0 || interfaces == ReadInterfaces::Row0To1 ||
           interfaces == ReadInterfaces::Row0To2 || interfaces == ReadInterfaces::Row0And2 || interfaces == ReadInterfaces::Row1And3;
}

constexpr bool is_valid(const DatumOverride override)
{
    return override == DatumOverride::None || override == DatumOverride::Zero;
}

constexpr bool is_valid(const DestAccess access)
{
    return access == DestAccess::RowMajor || access == DestAccess::Strided;
}

constexpr bool is_valid(const Concatenation concatenation)
{
    return concatenation == Concatenation::NewRow || concatenation == Concatenation::Append;
}

constexpr bool is_valid(const TileBoundary boundary)
{
    return boundary == TileBoundary::NotLast || boundary == TileBoundary::Last;
}

constexpr bool is_valid(const Transfer options)
{
    return hal::detail::is_valid(options.address_modifier, 2) && is_valid(options.interfaces) && is_valid(options.datum_override) &&
           is_valid(options.dest_access) && is_valid(options.concatenation) && is_valid(options.boundary);
}

inline constexpr __attribute__((always_inline)) void validate(const Transfer options)
{
    hal::detail::require(hal::detail::is_valid(options.address_modifier, 2), "PACR address modifier must be in [0, 3]");
    hal::detail::require(is_valid(options.interfaces), "PACR read interfaces must be a ReadInterfaces value");
    hal::detail::require(is_valid(options.datum_override), "PACR datum override must be None or Zero");
    hal::detail::require(is_valid(options.dest_access), "PACR Dst access must be RowMajor or Strided");
    hal::detail::require(is_valid(options.concatenation), "PACR concatenation must be NewRow or Append");
    hal::detail::require(is_valid(options.boundary), "PACR tile boundary must be NotLast or Last");
}

constexpr std::uint32_t transfer_operation(const Transfer options)
{
    return TT_OP_PACR(
        0 /*CfgContext*/,
        0 /*RowPadZero*/,
        hal::to_underlying(options.dest_access) /*DstAccessMode*/,
        options.address_modifier /*AddrMod*/,
        0 /*AddrCntContext*/,
        hal::to_underlying(options.datum_override) /*ZeroWrite*/,
        hal::to_underlying(options.interfaces) /*ReadIntfSel*/,
        0 /*OvrdThreadId*/,
        hal::to_underlying(options.concatenation) /*Concat*/,
        0 /*CtxtCtrl*/,
        0 /*Flush*/,
        hal::to_underlying(options.boundary) /*Last*/);
}
} // namespace detail

/** @brief Return whether data-transfer options can be encoded. */
constexpr bool is_valid(const Transfer options)
{
    return detail::is_valid(options);
}

/**
 * @brief Issue or encode the operations of the Blackhole packer.
 *
 * Blackhole has one packer with four Dst read interfaces. Template forms check their options at compile
 * time and issue one immediate instruction; runtime forms check them with LLK_ASSERT.
 *
 * @code
 * Packer0.transfer<Transfer {.interfaces = ReadInterfaces::Row0To1, .boundary = TileBoundary::Last}>();
 *
 * constexpr std::uint32_t pack_row = Packer0.operation().transfer<Transfer {.address_modifier = 1}>();
 * @endcode
 */
class Packer
{
public:
    /** @brief Encode the packer's operations without issuing them, for MOP or replay use. */
    class Operations
    {
    public:
        /**
         * @brief Encode a data transfer.
         *
         * @tparam Options: Data-transfer options.
         */
        template <Transfer Options>
        [[nodiscard]] constexpr hal::Operation transfer() const;

        /**
         * @brief Encode a runtime-selected data transfer.
         *
         * @param options: Data-transfer options.
         * @note Invalid options fail constant evaluation; at runtime, enable LLK assertions to diagnose them.
         */
        [[nodiscard]] constexpr hal::Operation transfer(Transfer options) const;
    };

    /** @brief Return an encoding view of the packer. */
    [[nodiscard]] constexpr Operations operation() const
    {
        return {};
    }

    /**
     * @brief Issue a data transfer.
     *
     * @tparam Options: Data-transfer options.
     */
    template <Transfer Options>
    inline __attribute__((always_inline)) void transfer() const
    {
        hal::detail::run_immediate<Operations {}.transfer<Options>()>();
    }

    /**
     * @brief Issue a runtime-selected data transfer.
     *
     * @param options: Data-transfer options.
     * @note Enable LLK assertions to diagnose invalid options; disabled assertions add no runtime check.
     */
    inline __attribute__((always_inline)) void transfer(const Transfer options) const
    {
        hal::run(operation().transfer(options));
    }
};

/** @brief The Blackhole packer. */
inline constexpr Packer Packer0 {};

template <Transfer Options>
constexpr hal::Operation Packer::Operations::transfer() const
{
    static_assert(is_valid(Options), "invalid PACR data-transfer options");
    return hal::detail::make_operation(detail::transfer_operation(Options));
}

constexpr hal::Operation Packer::Operations::transfer(const Transfer options) const
{
    detail::validate(options);
    return hal::detail::make_operation(detail::transfer_operation(options));
}

} // namespace hal::pack
