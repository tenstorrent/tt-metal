// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "utils/source.h"

namespace hal
{

namespace src_ops
{

/** @brief Select whether a fill touches only the current bank or both banks. */
enum class BankScope : std::uint8_t
{
    Current,
    All
};

/** @brief Select whose "current" bank a current-bank fill means. */
enum class BankOwner : std::uint8_t
{
    Unpacker, // current write bank (producer side)
    Math      // current read bank (consumer side)
};

/** @brief Select the datum pattern a fill writes. */
enum class Fill : std::uint8_t
{
    Zero,
    NegativeInfinity // all ones; GMPOOL identity
};

/** @brief Select whether a release also flips the math thread to the paired bank. */
enum class BankAdvance : std::uint8_t
{
    Flip,           // normal: hand the bank to the unpacker, math flips to the paired bank
    KeepReadingSame // give the bank away but keep reading it
};

/** @brief Select what a SrcB row shift feeds into datum 15. */
enum class ShiftFill : std::uint8_t
{
    Rotate, // datum 0 wraps around into datum 15
    Zero    // zero is injected into datum 15
};

/** @brief Select the direction of the combined SrcA column shift. */
enum class ShiftDirection : std::uint8_t
{
    TowardColumn0   = 0b11,
    AwayFromColumn0 = 0b10
};

} // namespace src_ops

/**
 * @brief Operations on the SrcA/SrcB register files as a resource.
 *
 * Instantiate through the @ref src variable template: `hal::src<hal::SrcA>`,
 * `hal::src<hal::SrcB>`, or `hal::src<hal::BothSources>`. Operations the hardware masks per
 * register (fill, release, publish, gating reset) accept any selection and issue one
 * instruction; single-register features (transpose, shifts, masks, rarefy) require exactly
 * that register's selection.
 *
 * @tparam Sources: Source registers every operation of this instance targets.
 */
template <SourceMask Sources>
struct SourceRegisters
{
private:
    static constexpr bool SINGLE_A           = Sources == SourceMask::SrcA;
    static constexpr bool SINGLE_B           = Sources == SourceMask::SrcB;
    static constexpr std::uint32_t MASK_BITS = source_bits(Sources);
    static constexpr std::uint32_t SRCB_ROWS = 64u;

    static_assert(MASK_BITS != 0 && MASK_BITS <= 3, "source selection must name SrcA, SrcB, or both");

public:
    // ---------------- Fill (ZEROSRC) ----------------

    /**
     * @brief Encode a fill of the selected source registers (ZEROSRC).
     *
     * @tparam Scope: Fill only the selected current bank or both banks.
     * @tparam Owner: Whose "current" bank a current-bank fill means.
     * @tparam Value: Datum pattern written into every filled register.
     */
    template <
        src_ops::BankScope Scope = src_ops::BankScope::Current,
        src_ops::BankOwner Owner = src_ops::BankOwner::Math,
        src_ops::Fill Value      = src_ops::Fill::Zero>
    static constexpr std::uint32_t fill_operation()
    {
        return TT_OP_ZEROSRC(
            Value == src_ops::Fill::NegativeInfinity ? 1u : 0u,
            Owner == src_ops::BankOwner::Math ? 1u : 0u,
            Scope == src_ops::BankScope::All ? 1u : 0u,
            MASK_BITS);
    }

    /** @brief Fill the selected source registers (ZEROSRC). */
    template <
        src_ops::BankScope Scope = src_ops::BankScope::Current,
        src_ops::BankOwner Owner = src_ops::BankOwner::Math,
        src_ops::Fill Value      = src_ops::Fill::Zero>
    static inline __attribute__((always_inline)) void fill()
    {
        INSTRUCTION_WORD((fill_operation<Scope, Owner, Value>()));
    }

    // ---------------- Data-valid handoff ----------------

    /**
     * @brief Encode a release of the selected current bank(s) back to the unpackers (CLEARDVALID).
     *
     * @tparam Advance: Whether the math thread also flips to the paired bank.
     * @note The full data-valid reset bit is not exposed: the ISA marks it unsupported on
     *       Blackhole. Hot loops should prefer releasing through the consuming operation
     *       (@ref hal::fpu SourceRelease, @ref hal::math_counters release) — zero extra
     *       instructions; this is the standalone form.
     */
    template <src_ops::BankAdvance Advance = src_ops::BankAdvance::Flip>
    static constexpr std::uint32_t release_operation()
    {
        return TT_OP_CLEARDVALID(MASK_BITS, Advance == src_ops::BankAdvance::KeepReadingSame ? 0b10u : 0u);
    }

    /** @brief Release the selected current bank(s) back to the unpackers (CLEARDVALID). */
    template <src_ops::BankAdvance Advance = src_ops::BankAdvance::Flip>
    static inline __attribute__((always_inline)) void release()
    {
        INSTRUCTION_WORD((release_operation<Advance>()));
    }

    /**
     * @brief Encode a publish of the unpacker-side write bank(s) to the math thread (SETDVALID).
     *
     * @note Constrained support on Blackhole: the ISA marks SETDVALID unsupported because the
     *       published bank's implied Src format becomes unpredictable. Existing call sites are
     *       math-filled banks (Dst-to-SrcB staging) with implied-format inference disabled. The
     *       sanctioned general-purpose publish is the unpacker-side handoff (UNPACR/UNPACR_NOP).
     */
    static constexpr std::uint32_t publish_operation()
    {
        return TT_OP_SETDVALID(MASK_BITS);
    }

    /** @brief Publish the unpacker-side write bank(s) to the math thread (SETDVALID). */
    static inline __attribute__((always_inline)) void publish()
    {
        INSTRUCTION_WORD(publish_operation());
    }

    // ---------------- Pipeline clock gating ----------------

    /** @brief Encode a reset of the selected source pipelines' clock gating to "don't gate" (GATESRCRST). */
    static constexpr std::uint32_t reset_pipeline_gating_operation()
    {
        return TT_OP_GATESRCRST((MASK_BITS >> 1) & 1u, MASK_BITS & 1u);
    }

    /** @brief Reset the selected source pipelines' clock gating to "don't gate" (GATESRCRST). */
    static inline __attribute__((always_inline)) void reset_pipeline_gating()
    {
        INSTRUCTION_WORD(reset_pipeline_gating_operation());
    }

    // ---------------- Single-register features ----------------

    /**
     * @brief Encode an in-place transpose of the selected register (TRNSPSRCA / TRNSPSRCB).
     *
     * SrcB transposes rows 16-31 of the current math bank: stage data into the upper half
     * first. Waits for data valid; does not use or advance the RWC.
     *
     * @note The SrcA form is absent from the ISA documentation and has no call sites —
     *       SrcA transposition is normally the unpacker's job. Validate on hardware before use.
     */
    static constexpr std::uint32_t transpose_operation()
    {
        static_assert(SINGLE_A || SINGLE_B, "transpose targets exactly one source register");
        return SINGLE_A ? TT_OP_TRNSPSRCA : TT_OP_TRNSPSRCB;
    }

    /** @brief Transpose the selected register in place (TRNSPSRCA / TRNSPSRCB). */
    static inline __attribute__((always_inline)) void transpose()
    {
        INSTRUCTION_WORD(transpose_operation());
    }

    /**
     * @brief Encode a one-datum combined column shift of all SrcA rows (SHIFTXA).
     *
     * @tparam Direction: Shift toward or away from column 0.
     * @note Blackhole erratum: the row-block selection reuses the last SrcA address, so the
     *       shifted block cannot be chosen; the ISA discourages use. No call sites exist.
     *       The hardware only supports a shift amount of one datum.
     */
    template <src_ops::ShiftDirection Direction>
    static constexpr std::uint32_t shift_columns_operation()
    {
        static_assert(SINGLE_A, "the combined column shift targets SrcA");
        return TT_OP_SHIFTXA(0u /* log2 amount: shift by one */, static_cast<std::uint32_t>(Direction));
    }

    /** @brief Shift all SrcA rows by one datum column (SHIFTXA). */
    template <src_ops::ShiftDirection Direction>
    static inline __attribute__((always_inline)) void shift_columns()
    {
        INSTRUCTION_WORD((shift_columns_operation<Direction>()));
    }

    /**
     * @brief Encode a one-datum left shift of one SrcB row (SHIFTXB).
     *
     * @tparam Feed: What enters datum 15 — the wrapped datum 0 or zero.
     * @tparam AddressMode: ADDR_MOD section applied after the shift.
     * @tparam Row: SrcB row to shift, added to the SrcB counter.
     */
    // Compile-time path
    template <std::uint32_t Row, src_ops::ShiftFill Feed = src_ops::ShiftFill::Zero, std::uint8_t AddressMode = 0>
    static constexpr std::uint32_t shift_row_operation()
    {
        static_assert(SINGLE_B, "the row shift targets SrcB");
        static_assert(Row < SRCB_ROWS, "SrcB row must be in [0, 63]");
        static_assert(AddressMode < 8u, "Blackhole address mode must be in [0, 7]");
        return TT_OP_SHIFTXB(AddressMode, Feed == src_ops::ShiftFill::Zero ? 1u : 0u, Row);
    }

    // Runtime path
    template <src_ops::ShiftFill Feed = src_ops::ShiftFill::Zero, std::uint8_t AddressMode = 0>
    static inline __attribute__((always_inline)) std::uint32_t shift_row_operation(const std::uint32_t row)
    {
        static_assert(SINGLE_B, "the row shift targets SrcB");
        static_assert(AddressMode < 8u, "Blackhole address mode must be in [0, 7]");
        LLK_ASSERT(row < SRCB_ROWS, "SrcB row must be in [0, 63]");
        return TT_OP_SHIFTXB(AddressMode, Feed == src_ops::ShiftFill::Zero ? 1u : 0u, row);
    }

    /** @brief Shift one SrcB row left by one datum (SHIFTXB). */
    // Compile-time path
    template <std::uint32_t Row, src_ops::ShiftFill Feed = src_ops::ShiftFill::Zero, std::uint8_t AddressMode = 0>
    static inline __attribute__((always_inline)) void shift_row()
    {
        INSTRUCTION_WORD((shift_row_operation<Row, Feed, AddressMode>()));
    }

    // Runtime path
    template <src_ops::ShiftFill Feed = src_ops::ShiftFill::Zero, std::uint8_t AddressMode = 0>
    static inline __attribute__((always_inline)) void shift_row(const std::uint32_t row)
    {
        ckernel::instrn_buffer[0] = shift_row_operation<Feed, AddressMode>(row);
    }

    /**
     * @brief Encode a rarefication of the SrcB registers (RAREB).
     *
     * @note Legacy operation: absent from the ISA documentation, no call sites. Validate on
     *       hardware before use.
     */
    static constexpr std::uint32_t rarefy_operation()
    {
        static_assert(SINGLE_B, "rarefication targets SrcB");
        return TT_OP_RAREB;
    }

    /** @brief Rarefy the SrcB registers (RAREB). */
    static inline __attribute__((always_inline)) void rarefy()
    {
        INSTRUCTION_WORD(rarefy_operation());
    }

    /**
     * @brief Encode one SrcA right-shift mask write (SETASHRMH / SETASHRMH0 / SETASHRMH1 / SETASHRMV).
     *
     * @tparam RegisterMask: 16-bit register mask (20 bits for the vertical mask).
     * @tparam HaloMask: Left-halo mask bit; only the plain and thread0 horizontal forms use it.
     * @note Grayskull-era convolution-halo legacy: absent from the ISA documentation, no call
     *       sites. Wrapped for encoding completeness only; validate on hardware before use.
     */
    template <std::uint32_t RegisterMask, std::uint32_t HaloMask = 0>
    static constexpr std::uint32_t right_shift_mask_horizontal_operation()
    {
        static_assert(SINGLE_A, "the right-shift masks target SrcA");
        static_assert(RegisterMask < (1u << 16), "horizontal right-shift register mask is 16 bits");
        static_assert(HaloMask < 2u, "halo mask is one bit");
        return TT_OP_SETASHRMH(RegisterMask, HaloMask);
    }

    template <std::uint32_t RegisterMask, std::uint32_t HaloMask = 0>
    static constexpr std::uint32_t right_shift_mask_horizontal_thread0_operation()
    {
        static_assert(SINGLE_A, "the right-shift masks target SrcA");
        static_assert(RegisterMask < (1u << 16), "horizontal right-shift register mask is 16 bits");
        static_assert(HaloMask < 2u, "halo mask is one bit");
        return TT_OP_SETASHRMH0(RegisterMask, HaloMask);
    }

    template <std::uint32_t RegisterMask>
    static constexpr std::uint32_t right_shift_mask_horizontal_thread1_operation()
    {
        static_assert(SINGLE_A, "the right-shift masks target SrcA");
        static_assert(RegisterMask < (1u << 16), "horizontal right-shift register mask is 16 bits");
        return TT_OP_SETASHRMH1(RegisterMask, 0u /* no thread1 left halo */);
    }

    template <std::uint32_t RegisterMask>
    static constexpr std::uint32_t right_shift_mask_vertical_operation()
    {
        static_assert(SINGLE_A, "the right-shift masks target SrcA");
        static_assert(RegisterMask < (1u << 20), "vertical right-shift register mask is 20 bits");
        return TT_OP_SETASHRMV(RegisterMask);
    }

    /** @brief Write one SrcA right-shift mask (SETASHRMH / SETASHRMH0 / SETASHRMH1 / SETASHRMV). */
    template <std::uint32_t RegisterMask, std::uint32_t HaloMask = 0>
    static inline __attribute__((always_inline)) void set_right_shift_mask_horizontal()
    {
        INSTRUCTION_WORD((right_shift_mask_horizontal_operation<RegisterMask, HaloMask>()));
    }

    template <std::uint32_t RegisterMask, std::uint32_t HaloMask = 0>
    static inline __attribute__((always_inline)) void set_right_shift_mask_horizontal_thread0()
    {
        INSTRUCTION_WORD((right_shift_mask_horizontal_thread0_operation<RegisterMask, HaloMask>()));
    }

    template <std::uint32_t RegisterMask>
    static inline __attribute__((always_inline)) void set_right_shift_mask_horizontal_thread1()
    {
        INSTRUCTION_WORD((right_shift_mask_horizontal_thread1_operation<RegisterMask>()));
    }

    template <std::uint32_t RegisterMask>
    static inline __attribute__((always_inline)) void set_right_shift_mask_vertical()
    {
        INSTRUCTION_WORD((right_shift_mask_vertical_operation<RegisterMask>()));
    }
};

/**
 * @brief Per-selection entry point for source-register operations.
 *
 * `hal::src<hal::SrcA>`, `hal::src<hal::SrcB>`, and `hal::src<hal::BothSources>` expose the
 * SrcA/SrcB resource features: fill, data-valid handoff, gating reset, transpose, shifts,
 * shift masks, and rarefication.
 */
template <SourceMask Sources>
inline constexpr SourceRegisters<Sources> src {};

} // namespace hal
