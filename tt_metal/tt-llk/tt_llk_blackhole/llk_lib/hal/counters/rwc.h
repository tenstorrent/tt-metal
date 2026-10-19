// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file
 * @brief Describe, encode, and issue Blackhole register-word-counter updates.
 */

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"

/**
 * @brief Register-word counters (RWCs) of the issuing math thread.
 *
 * Each of SrcA, SrcB, and Dest has a live counter X and a carry (CR) shadow X_Cr. Every update
 * is one of the operations below, applied independently per counter; the HAL selects SETRWC
 * and/or INCRWC to encode them.
 */
namespace hal::counters::rwc
{

/** @brief Assign a counter and its CR shadow: X = X_Cr = value. */
class Set
{
public:
    std::uint32_t value; ///< Assignment in [0, 15].

    constexpr explicit __attribute__((always_inline)) Set(const std::uint32_t value) : value(value)
    {
    }
};

/** @brief Advance only the live counter: X += value, preserving X_Cr. */
class Increment
{
public:
    std::uint32_t value; ///< Increment in [0, 15].

    constexpr explicit __attribute__((always_inline)) Increment(const std::uint32_t value) : value(value)
    {
    }
};

/** @brief Advance the CR shadow and reload the live counter from it: X_Cr += value, X = X_Cr. Zero rewinds X to X_Cr. */
class AdvanceAndReset
{
public:
    std::uint32_t value; ///< Increment in [0, 15].

    constexpr explicit __attribute__((always_inline)) AdvanceAndReset(const std::uint32_t value = 0) : value(value)
    {
    }
};

/** @brief Dest only. Advance the live counter and save it to the CR shadow: X += value, X_Cr = X. */
class IncrementAndSave
{
public:
    std::uint32_t value; ///< Increment in [0, 15].

    constexpr explicit __attribute__((always_inline)) IncrementAndSave(const std::uint32_t value = 0) : value(value)
    {
    }
};

/** @brief Operation applied to one counter. */
enum class Op : std::uint8_t
{
    None,            ///< Counter is unchanged.
    Set,             ///< See @ref Set.
    Increment,       ///< See @ref Increment.
    AdvanceAndReset, ///< See @ref AdvanceAndReset.
    IncrementAndSave ///< See @ref IncrementAndSave.
};

/** @brief Optional SrcA or SrcB update; omission leaves the counter unchanged. */
class SrcUpdate
{
public:
    Op op               = Op::None; ///< Operation applied to the counter.
    std::uint32_t value = 0;        ///< Operand of op.

    constexpr SrcUpdate() = default;

    constexpr __attribute__((always_inline)) SrcUpdate(const Set update) : op(Op::Set), value(update.value)
    {
    }

    constexpr __attribute__((always_inline)) SrcUpdate(const Increment update) : op(Op::Increment), value(update.value)
    {
    }

    constexpr __attribute__((always_inline)) SrcUpdate(const AdvanceAndReset update) : op(Op::AdvanceAndReset), value(update.value)
    {
    }
};

/** @brief Optional Dest update; omission leaves the counter unchanged. */
class DestUpdate : public SrcUpdate
{
public:
    using SrcUpdate::SrcUpdate;

    constexpr DestUpdate() = default;

    constexpr __attribute__((always_inline)) DestUpdate(const IncrementAndSave update)
    {
        op    = Op::IncrementAndSave;
        value = update.value;
    }
};

/** @brief Select source operands whose banks SETRWC flips. */
enum class Src : std::uint8_t
{
    None = 0x0, ///< No flip.
    A    = 0x1, ///< SrcA.
    B    = 0x2  ///< SrcB.
};

inline constexpr Src operator|(const Src lhs, const Src rhs)
{
    return static_cast<Src>(static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs));
}

/**
 * @brief Describe a math-thread register-word-counter update independently of its instruction encoding.
 *
 * Set, IncrementAndSave, flip, and clear_fidelity need SETRWC; Increment needs INCRWC.
 * AdvanceAndReset joins SETRWC when one is issued, otherwise INCRWC. Updates needing both
 * instructions issue SETRWC first; the counters they touch are disjoint.
 *
 * @code
 * constexpr MathCounter next_face = MathCounter {.src_b = Set(0), .dest = AdvanceAndReset(8), .flip = Src::A | Src::B};
 * set<next_face>();
 *
 * constexpr auto next_rows = MathCounter {.dest = Increment(8)}.operation();
 * @endcode
 */
class MathCounter
{
public:
    SrcUpdate src_a {}; ///< SrcA update; omitted by default.
    SrcUpdate src_b {}; ///< SrcB update; omitted by default.
    DestUpdate dest {}; ///< Dest update; omitted by default.

    /**
     * @brief Source banks to flip while issuing SETRWC.
     *
     * Math switches to each selected source's paired bank and, unless the per-thread
     * CLR_DVALID_Src*_Disable configuration is set, returns the consumed bank to the unpacker.
     */
    Src flip = Src::None;

    bool clear_fidelity = false; ///< Reset the fidelity phase to zero while issuing SETRWC.

    /**
     * @brief Encode the update as one instruction without issuing it.
     *
     * @return The encoded SETRWC or INCRWC instruction.
     * @note Do not combine Increment with Set, IncrementAndSave, flip, or clear_fidelity; issue such
     *       updates with set(). Invalid descriptors fail constant evaluation, or LLK_ASSERT at runtime.
     */
    [[nodiscard]] constexpr std::uint32_t operation() const;
};

namespace detail
{

inline constexpr std::uint32_t RwcValueMax = 0xf; ///< Four-bit SETRWC/INCRWC immediate.

inline __attribute__((always_inline)) constexpr bool any_op(const MathCounter& descriptor, const Op op)
{
    return descriptor.src_a.op == op || descriptor.src_b.op == op || descriptor.dest.op == op;
}

/** @brief Test whether the update needs SETRWC. */
inline __attribute__((always_inline)) constexpr bool needs_setrwc(const MathCounter& descriptor)
{
    return any_op(descriptor, Op::Set) || any_op(descriptor, Op::IncrementAndSave) || descriptor.flip != Src::None || descriptor.clear_fidelity;
}

/** @brief Test whether the update needs INCRWC. */
inline __attribute__((always_inline)) constexpr bool needs_incrwc(const MathCounter& descriptor)
{
    return any_op(descriptor, Op::Increment) || (!needs_setrwc(descriptor) && any_op(descriptor, Op::AdvanceAndReset));
}

/** @brief Test whether the update is nonempty; ranges are checked by the encoders. */
inline __attribute__((always_inline)) constexpr bool is_valid(const MathCounter& descriptor)
{
    return needs_setrwc(descriptor) || needs_incrwc(descriptor);
}

/**
 * @brief Fail constant evaluation, or LLK_ASSERT at runtime, when a condition does not hold.
 *
 * @param valid: Condition required by the descriptor or instruction encoding.
 */
inline __attribute__((always_inline)) constexpr void require(const bool valid)
{
    if (!valid)
    {
        if (__builtin_is_constant_evaluated())
        {
            __builtin_trap();
        }
        else
        {
            // Keep assertion assembly outside the C++17 constexpr body.
            [valid] { LLK_ASSERT(valid, "invalid math-counter descriptor or value outside the instruction's four-bit field"); }();
        }
    }
}

/**
 * @brief Encode one counter's share of an instruction.
 *
 * @param update: Counter update.
 * @param in_setrwc: True to encode for SETRWC, false for INCRWC.
 * @param bit: Counter's bit in the instruction's select and CR masks.
 * @param select: Accumulates the SETRWC write mask.
 * @param carry: Accumulates the CR mask.
 * @return The counter's immediate, or zero if the instruction leaves it unchanged.
 */
inline __attribute__((always_inline)) constexpr std::uint32_t encode_counter(
    const SrcUpdate& update, const bool in_setrwc, const std::uint32_t bit, std::uint32_t& select, std::uint32_t& carry)
{
    const bool encoded = in_setrwc ? update.op == Op::Set || update.op == Op::AdvanceAndReset || update.op == Op::IncrementAndSave
                                   : update.op == Op::Increment || update.op == Op::AdvanceAndReset;
    if (!encoded)
    {
        return 0;
    }
    require(update.value <= RwcValueMax);
    select |= bit;
    if (update.op == Op::AdvanceAndReset)
    {
        carry |= bit;
    }
    else if (update.op == Op::IncrementAndSave)
    {
        carry |= ckernel::p_setrwc::C_TO_CR_MODE;
    }
    return update.value;
}

/** @brief Encode the SETRWC share of a valid update. */
inline __attribute__((always_inline)) constexpr std::uint32_t setrwc(const MathCounter& descriptor)
{
    std::uint32_t select = 0;
    std::uint32_t carry  = 0;
    const auto d         = encode_counter(descriptor.dest, true, ckernel::p_setrwc::SET_D, select, carry);
    const auto b         = encode_counter(descriptor.src_b, true, ckernel::p_setrwc::SET_B, select, carry);
    const auto a         = encode_counter(descriptor.src_a, true, ckernel::p_setrwc::SET_A, select, carry);
    if (descriptor.clear_fidelity)
    {
        select |= ckernel::p_setrwc::SET_F;
    }
    return TT_OP_SETRWC(static_cast<std::uint32_t>(descriptor.flip), carry, d, b, a, select);
}

/** @brief Encode the INCRWC share of a valid update. */
inline __attribute__((always_inline)) constexpr std::uint32_t incrwc(const MathCounter& descriptor)
{
    // INCRWC has no write mask: omitted counters encode a zero increment without CR.
    std::uint32_t select = 0;
    std::uint32_t carry  = 0;
    const auto d         = encode_counter(descriptor.dest, false, ckernel::p_setrwc::CR_D, select, carry);
    const auto b         = encode_counter(descriptor.src_b, false, ckernel::p_setrwc::CR_B, select, carry);
    const auto a         = encode_counter(descriptor.src_a, false, ckernel::p_setrwc::CR_A, select, carry);
    return TT_OP_INCRWC(carry, d, b, a);
}

} // namespace detail

inline constexpr __attribute__((always_inline)) std::uint32_t MathCounter::operation() const
{
    const bool setrwc = detail::needs_setrwc(*this);
    detail::require(setrwc != detail::needs_incrwc(*this));
    return setrwc ? detail::setrwc(*this) : detail::incrwc(*this);
}

/**
 * @brief Issue a compile-time counter update as immediate instructions.
 *
 * @tparam Descriptor: Counter update with values in [0, 15], rejected at compile time when invalid.
 * @note Issues SETRWC, INCRWC, or SETRWC followed by INCRWC.
 */
template <auto Descriptor>
inline __attribute__((always_inline)) void set()
{
    static_assert(detail::is_valid(Descriptor), "math-counter descriptor must update a counter, flip a bank, or clear the fidelity phase");
    if constexpr (detail::needs_setrwc(Descriptor))
    {
        constexpr std::uint32_t word = detail::setrwc(Descriptor);
        TTI_INSN(word);
    }
    if constexpr (detail::needs_incrwc(Descriptor))
    {
        constexpr std::uint32_t word = detail::incrwc(Descriptor);
        TTI_INSN(word);
    }
}

/**
 * @brief Issue a counter update with runtime values through the instruction buffer.
 *
 * @param descriptor: Counter update with values in [0, 15].
 * @note Keep operations, flip, and clear_fidelity constant so instruction selection folds away;
 *       only the values need be runtime. Validation uses LLK_ASSERT; the preconditions still
 *       apply when assertions are disabled.
 */
inline __attribute__((always_inline)) void set(const MathCounter descriptor)
{
    detail::require(detail::is_valid(descriptor));
    if (detail::needs_setrwc(descriptor))
    {
        TT_INSN(detail::setrwc(descriptor));
    }
    if (detail::needs_incrwc(descriptor))
    {
        TT_INSN(detail::incrwc(descriptor));
    }
}

} // namespace hal::counters::rwc
