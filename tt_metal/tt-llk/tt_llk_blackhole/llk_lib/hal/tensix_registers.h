// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file
 * @brief Operations on the Tensix register files: SrcA, SrcB and Dst.
 *
 * @see hal::tensix_registers::SrcA
 * @see hal::tensix_registers::SrcB
 * @see hal::tensix_registers::Dst
 */

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "utils/helpers.h"
#include "utils/operation.h"
#include "utils/validate.h"

namespace hal::tensix_registers
{

namespace detail
{
/** @brief {SrcB, SrcA} selection mask of a source-register operation. */
enum class Sources : std::uint8_t
{
    A    = 0b01,
    B    = 0b10,
    Both = 0b11
};

constexpr Sources operator|(const Sources lhs, const Sources rhs)
{
    return static_cast<Sources>(hal::to_underlying(lhs) | hal::to_underlying(rhs));
}
} // namespace detail

/** @brief Select whether a release also moves the Matrix Unit to the other bank. */
enum class BankAdvance : bool
{
    Flip            = false, ///< Give the bank to the unpacker and read the other bank next.
    KeepReadingSame = true   ///< Give the bank to the unpacker but keep reading it.
};

/**
 * @brief Issue or encode the operations of SrcA, SrcB, or both.
 *
 * Use @ref SrcA, @ref SrcB, or `SrcA | SrcB`, which issues one instruction for both registers.
 * Every operation is encoded at compile time and issued as one immediate instruction.
 *
 * @code
 * SrcB.transpose();
 * (SrcA | SrcB).release();
 *
 * constexpr std::uint32_t zero_srca = SrcA.operation().zero_out();
 * @endcode
 *
 * @tparam Selection: Source registers every operation targets.
 */
template <detail::Sources Selection>
class SourceRegister
{
public:
    /** @brief Encode the operations without issuing them, for MOP or replay use. */
    class Operations
    {
    public:
        /**
         * @brief Encode writing zeros into the Matrix Unit's current bank (ZEROSRC).
         *
         * @note Zeroing the unpacker's bank, zeroing to negative infinity, and the both-banks form
         *       other than for both registers are UnsupportedFunctionality on Blackhole.
         */
        [[nodiscard]] constexpr hal::Operation zero_out() const
        {
            return hal::detail::make_operation(
                TT_OP_ZEROSRC(0 /*zero_val*/, 1 /*write_mode: Matrix Unit bank*/, 0 /*bank_mask*/, hal::to_underlying(Selection) /*src_mask*/));
        }

        /**
         * @brief Encode a release of the Matrix Unit's current bank to the unpacker (CLEARDVALID).
         *
         * @tparam Advance: Whether the Matrix Unit moves to the other bank.
         * @note The data-valid reset is not exposed: it is UnsupportedFunctionality on Blackhole.
         */
        template <BankAdvance Advance = BankAdvance::Flip>
        [[nodiscard]] constexpr hal::Operation release() const
        {
            return hal::detail::make_operation(
                TT_OP_CLEARDVALID(hal::to_underlying(Selection) /*cleardvalid*/, (hal::to_underlying(Advance) ? 0b10u : 0u) /*reset*/));
        }

        /**
         * @brief Encode an in-place transpose of rows 16 to 31 of SrcB's current Matrix Unit bank (TRNSPSRCB).
         *
         * Waits for data valid; does not use or advance the RWC.
         */
        [[nodiscard]] constexpr hal::Operation transpose() const
        {
            static_assert(Selection == detail::Sources::B, "transpose targets SrcB");
            return hal::detail::make_operation(TT_OP_TRNSPSRCB);
        }

        /**
         * @brief Encode a reset of the SrcA and SrcB gate control in front of the Matrix Unit (GATESRCRST).
         *
         * @note Blackhole does not document GATESRCRST. Wormhole documents only the SrcB operand, as an
         *       invalidation of the SrcB operand cache; this is production's form with both operands set.
         */
        [[nodiscard]] constexpr hal::Operation reset_gate_control() const
        {
            static_assert(Selection == detail::Sources::Both, "GATESRCRST resets both gate controls: issue it on SrcA | SrcB");
            return hal::detail::make_operation(TT_OP_GATESRCRST(1 /*reset_srcb_gate_control*/, 1 /*reset_srca_gate_control*/));
        }
    };

    /** @brief Return an encoding view of the selected registers. */
    [[nodiscard]] constexpr Operations operation() const
    {
        return {};
    }

    /** @brief Write zeros into the Matrix Unit's current bank (ZEROSRC). */
    inline __attribute__((always_inline)) void zero_out() const
    {
        hal::detail::run_immediate<Operations {}.zero_out()>();
    }

    /**
     * @brief Release the Matrix Unit's current bank to the unpacker (CLEARDVALID).
     *
     * @tparam Advance: Whether the Matrix Unit moves to the other bank.
     * @note Hot loops should prefer releasing through the consuming FPU operation or an RWC flip.
     */
    template <BankAdvance Advance = BankAdvance::Flip>
    inline __attribute__((always_inline)) void release() const
    {
        hal::detail::run_immediate<Operations {}.template release<Advance>()>();
    }

    /** @brief Transpose rows 16 to 31 of SrcB's current Matrix Unit bank in place (TRNSPSRCB). */
    inline __attribute__((always_inline)) void transpose() const
    {
        hal::detail::run_immediate<Operations {}.transpose()>();
    }

    /** @brief Reset the SrcA and SrcB gate control in front of the Matrix Unit (GATESRCRST). */
    inline __attribute__((always_inline)) void reset_gate_control() const
    {
        hal::detail::run_immediate<Operations {}.reset_gate_control()>();
    }
};

/** @brief Select both source registers for one instruction. */
template <detail::Sources Lhs, detail::Sources Rhs>
constexpr SourceRegister<Lhs | Rhs> operator|(SourceRegister<Lhs>, SourceRegister<Rhs>)
{
    return {};
}

/** @brief Source register A, written by unpacker 0 and read by the Matrix Unit. */
inline constexpr SourceRegister<detail::Sources::A> SrcA {};

/** @brief Source register B, written by unpacker 1 and read by the Matrix Unit. */
inline constexpr SourceRegister<detail::Sources::B> SrcB {};

/** @brief Select how many Dst rows a ZEROACC marks; the value is the ZEROACC mode. */
enum class ZeroScope : std::uint8_t
{
    SingleRow = 0, ///< One row: index plus the Dst RWC and configured base offsets.
    Face      = 1, ///< 16 rows of face index, relative to the addressed half of Dst.
    Half      = 2, ///< One half of Dst; index 0 or 1.
    All       = 3  ///< All of Dst; index 0.
};

/** @brief Select 16-bit or 32-bit Dst row addressing. */
enum class DestWidth : bool
{
    Bits16 = false,
    Bits32 = true
};

/**
 * @brief Describe one ZEROACC that marks Dst rows undefined (clears their DstRowValid).
 *
 * Undefined rows read as zero to the packer and as the instruction's identity to the Matrix Unit.
 */
class DstMarkInvalid
{
public:
    ZeroScope scope;                               ///< Rows marked undefined.
    std::uint32_t index       = 0;                 ///< Row, face or half, by scope.
    DestWidth width           = DestWidth::Bits16; ///< Row addressing.
    std::uint8_t address_mode = 0;                 ///< ADDR_MOD section applied after the operation, in [0, 7].
};

/**
 * @brief Describe one ZEROACC that sets the valid bits (DstRowValid) of a Dst face again (Revert).
 *
 * After an unpack of 32-bit data to Dst, this restores valid bits that the unpack's own update may
 * have lost to a ZEROACC in the same cycle.
 *
 * @note Only DestWidth::Bits32 is contractual. With DestWidth::Bits16 the ISA calls the effect
 *       NonContractualBehavior; current silicon marks the face valid as well.
 */
class DstMarkValid
{
public:
    std::uint32_t face        = 0;                 ///< Face, relative to the addressed half of Dst.
    DestWidth width           = DestWidth::Bits32; ///< Row addressing.
    std::uint8_t address_mode = 0;                 ///< ADDR_MOD section applied after the operation, in [0, 7].
};

namespace detail
{
// ZEROACC's row, face or half selector is the 10-bit Imm10.
inline constexpr std::uint8_t INDEX_WIDTH = 10;

constexpr bool is_valid(const ZeroScope scope)
{
    return scope == ZeroScope::SingleRow || scope == ZeroScope::Face || scope == ZeroScope::Half || scope == ZeroScope::All;
}

constexpr bool is_valid(const DestWidth width)
{
    return width == DestWidth::Bits16 || width == DestWidth::Bits32;
}

constexpr bool is_valid_index(const ZeroScope scope, const std::uint32_t index)
{
    switch (scope)
    {
        case ZeroScope::Half:
            return index < 2u;
        case ZeroScope::All:
            return index == 0u;
        default:
            return hal::detail::is_valid(index, INDEX_WIDTH);
    }
}

constexpr bool is_valid(const DstMarkInvalid options)
{
    return is_valid(options.scope) && is_valid_index(options.scope, options.index) && is_valid(options.width) && hal::detail::is_valid(options.address_mode, 3);
}

constexpr bool is_valid(const DstMarkValid options)
{
    return hal::detail::is_valid(options.face, INDEX_WIDTH) && is_valid(options.width) && hal::detail::is_valid(options.address_mode, 3);
}

inline constexpr __attribute__((always_inline)) void validate(const DstMarkInvalid options)
{
    hal::detail::require(is_valid(options.scope), "ZEROACC scope must be SingleRow, Face, Half, or All");
    hal::detail::require(is_valid_index(options.scope, options.index), "ZEROACC index must be in [0, 1023], 0 or 1 for Half, and 0 for All");
    hal::detail::require(is_valid(options.width), "ZEROACC Dst width must be Bits16 or Bits32");
    hal::detail::require(hal::detail::is_valid(options.address_mode, 3), "Blackhole ZEROACC address mode must be in [0, 7]");
}

inline constexpr __attribute__((always_inline)) void validate(const DstMarkValid options)
{
    hal::detail::require(hal::detail::is_valid(options.face, INDEX_WIDTH), "ZEROACC face must be in [0, 1023]");
    hal::detail::require(is_valid(options.width), "ZEROACC Dst width must be Bits16 or Bits32");
    hal::detail::require(hal::detail::is_valid(options.address_mode, 3), "Blackhole ZEROACC address mode must be in [0, 7]");
}

constexpr std::uint32_t mark_invalid_operation(const DstMarkInvalid options)
{
    return TT_OP_ZEROACC(
        static_cast<std::uint32_t>(options.scope) /*clear_mode*/,
        static_cast<std::uint32_t>(options.width) /*use_32_bit_mode*/,
        0u /*clear_zero_flags*/,
        options.address_mode /*addr_mode*/,
        options.index /*where*/);
}

constexpr std::uint32_t mark_valid_operation(const DstMarkValid options)
{
    return TT_OP_ZEROACC(
        static_cast<std::uint32_t>(ZeroScope::Face) /*clear_mode*/,
        static_cast<std::uint32_t>(options.width) /*use_32_bit_mode*/,
        1u /*clear_zero_flags: Revert*/,
        options.address_mode /*addr_mode*/,
        options.face /*where*/);
}
} // namespace detail

/** @brief Return whether Dst mark-invalid options can be encoded. */
constexpr bool is_valid(const DstMarkInvalid options)
{
    return detail::is_valid(options);
}

/** @brief Return whether Dst mark-valid options can be encoded. */
constexpr bool is_valid(const DstMarkValid options)
{
    return detail::is_valid(options);
}

/**
 * @brief Issue or encode the operations of Dst.
 *
 * Template forms check their options at compile time and issue one immediate instruction; runtime
 * forms check them with LLK_ASSERT.
 *
 * @code
 * Dst.mark_invalid<DstMarkInvalid {.scope = ZeroScope::All, .width = DestWidth::Bits32, .address_mode = 1}>();
 * Dst.mark_invalid({.scope = ZeroScope::Half, .index = dest_offset_id, .address_mode = 1});
 * @endcode
 */
class DestRegister
{
public:
    /** @brief Encode Dst operations without issuing them, for MOP or replay use. */
    class Operations
    {
    public:
        /**
         * @brief Encode marking Dst rows invalid (ZEROACC).
         *
         * @tparam Options: Rows, width and address mode.
         */
        template <DstMarkInvalid Options>
        [[nodiscard]] constexpr hal::Operation mark_invalid() const
        {
            static_assert(is_valid(Options), "invalid ZEROACC mark-invalid options");
            return hal::detail::make_operation(detail::mark_invalid_operation(Options));
        }

        /**
         * @brief Encode marking runtime-selected Dst rows invalid (ZEROACC).
         *
         * @param options: Rows, width and address mode.
         * @note Invalid options fail constant evaluation; at runtime, enable LLK assertions to diagnose them.
         */
        [[nodiscard]] constexpr hal::Operation mark_invalid(const DstMarkInvalid options) const
        {
            detail::validate(options);
            return hal::detail::make_operation(detail::mark_invalid_operation(options));
        }

        /**
         * @brief Encode a mark-valid of one face (ZEROACC Revert).
         *
         * @tparam Options: Face, width and address mode.
         */
        template <DstMarkValid Options>
        [[nodiscard]] constexpr hal::Operation mark_valid() const
        {
            static_assert(is_valid(Options), "invalid ZEROACC mark-valid options");
            return hal::detail::make_operation(detail::mark_valid_operation(Options));
        }

        /**
         * @brief Encode a runtime-selected mark-valid of one face (ZEROACC Revert).
         *
         * @param options: Face, width and address mode.
         * @note Invalid options fail constant evaluation; at runtime, enable LLK assertions to diagnose them.
         */
        [[nodiscard]] constexpr hal::Operation mark_valid(const DstMarkValid options) const
        {
            detail::validate(options);
            return hal::detail::make_operation(detail::mark_valid_operation(options));
        }
    };

    /** @brief Return an encoding view of Dst. */
    [[nodiscard]] constexpr Operations operation() const
    {
        return {};
    }

    /**
     * @brief Mark Dst rows invalid (ZEROACC); the packer then reads them as zero.
     *
     * @tparam Options: Rows, width and address mode.
     */
    template <DstMarkInvalid Options>
    inline __attribute__((always_inline)) void mark_invalid() const
    {
        hal::detail::run_immediate<Operations {}.mark_invalid<Options>()>();
    }

    /**
     * @brief Mark runtime-selected Dst rows invalid (ZEROACC); the packer then reads them as zero.
     *
     * @param options: Rows, width and address mode.
     * @note Enable LLK assertions to diagnose invalid options; disabled assertions add no runtime check.
     */
    inline __attribute__((always_inline)) void mark_invalid(const DstMarkInvalid options) const
    {
        hal::run(operation().mark_invalid(options));
    }

    /**
     * @brief Mark one Dst face valid again (ZEROACC Revert).
     *
     * @tparam Options: Face, width and address mode.
     */
    template <DstMarkValid Options>
    inline __attribute__((always_inline)) void mark_valid() const
    {
        hal::detail::run_immediate<Operations {}.mark_valid<Options>()>();
    }

    /**
     * @brief Mark one runtime-selected Dst face valid again (ZEROACC Revert).
     *
     * @param options: Face, width and address mode.
     * @note Enable LLK assertions to diagnose invalid options; disabled assertions add no runtime check.
     */
    inline __attribute__((always_inline)) void mark_valid(const DstMarkValid options) const
    {
        hal::run(operation().mark_valid(options));
    }
};

/** @brief The destination register. */
inline constexpr DestRegister Dst {};

} // namespace hal::tensix_registers
