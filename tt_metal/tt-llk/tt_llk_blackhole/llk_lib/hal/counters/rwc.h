// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file
 * @brief Describe, encode, and issue Blackhole register-word-counter updates.
 */

#pragma once

#include <cstdint>

#include "../utils/helpers.h"
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

/** @brief Operation applied to one counter. */
enum class Op : std::uint8_t
{
    None,            ///< Counter is unchanged.
    Set,             ///< See @ref Set.
    Increment,       ///< See @ref Increment.
    AdvanceAndReset, ///< See @ref AdvanceAndReset.
    IncrementAndSave ///< See @ref IncrementAndSave.
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
    return static_cast<Src>(hal::to_underlying(lhs) | hal::to_underlying(rhs));
}

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

/**
 * @brief Describe a math-thread register-word-counter update independently of its instruction encoding.
 *
 * Set, IncrementAndSave, flip, and clear_fidelity need SETRWC; Increment needs INCRWC; AdvanceAndReset
 * fits either. The HAL issues the instructions the fields require, SETRWC first, and encodes each
 * field in exactly one of them; see detail::plan() and detail::owner().
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
     * @note Do not combine Increment with Set, IncrementAndSave, flip, or clear_fidelity; such updates
     *       need SETRWC and INCRWC, so issue them with set(). Invalid descriptors fail constant
     *       evaluation, or LLK_ASSERT at runtime.
     */
    [[nodiscard]] constexpr std::uint32_t operation() const;
};

namespace detail
{

inline constexpr std::uint32_t RwcValueMax = 0xf; ///< Four-bit SETRWC/INCRWC immediate.

/** @brief Set of RWC instructions, one bit per instruction. */
enum class Instr : std::uint8_t
{
    None   = 0x0, ///< No instruction.
    Setrwc = 0x1, ///< SETRWC.
    Incrwc = 0x2  ///< INCRWC.
};

inline __attribute__((always_inline)) constexpr Instr operator|(const Instr lhs, const Instr rhs)
{
    return static_cast<Instr>(hal::to_underlying(lhs) | hal::to_underlying(rhs));
}

inline __attribute__((always_inline)) constexpr Instr operator&(const Instr lhs, const Instr rhs)
{
    return static_cast<Instr>(hal::to_underlying(lhs) & hal::to_underlying(rhs));
}

/** @brief Test whether a set of instructions contains instr. */
inline __attribute__((always_inline)) constexpr bool issues(const Instr instrs, const Instr instr)
{
    return (instrs & instr) != Instr::None;
}

/** @brief SETRWC write-mask bits. */
enum class WriteBit : std::uint32_t
{
    SrcA     = 0x1, ///< Write SrcA.
    SrcB     = 0x2, ///< Write SrcB.
    Dest     = 0x4, ///< Write Dest.
    Fidelity = 0x8  ///< Clear the fidelity phase.
};

/** @brief SETRWC and INCRWC CR-mask bits. */
enum class CrBit : std::uint32_t
{
    SrcA      = 0x1, ///< Apply the SrcA value relative to its CR shadow.
    SrcB      = 0x2, ///< Apply the SrcB value relative to its CR shadow.
    Dest      = 0x4, ///< Apply the Dest value relative to its CR shadow.
    DestCToCr = 0x8  ///< SETRWC only: apply the Dest value to the live counter and save it to the CR shadow.
};

/** @brief MathCounter fields that are not counter updates. */
enum class Control : std::uint8_t
{
    Flip,         ///< Source-bank flip.
    ClearFidelity ///< Fidelity-phase clear.
};

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
 * @brief Reject MathCounter::operation() for a descriptor that needs both SETRWC and INCRWC.
 *
 * Not constexpr, so constant evaluation fails with an error naming this function.
 */
inline void reject_two_instructions()
{
    LLK_ASSERT(false, "MathCounter::operation() encodes one instruction, but this descriptor needs SETRWC and INCRWC; issue it with counters::set()");
}

/** @brief Instructions able to encode an operation. Edit this table for a different instruction set. */
inline __attribute__((always_inline)) constexpr Instr encodable_in(const Op op)
{
    switch (op)
    {
        case Op::Set:
        case Op::IncrementAndSave:
            return Instr::Setrwc;
        case Op::Increment:
            return Instr::Incrwc;
        case Op::AdvanceAndReset:
            return Instr::Setrwc | Instr::Incrwc;
        case Op::None:
            break;
    }
    return Instr::None;
}

/** @brief Instructions able to encode a control. Edit this table for a different instruction set. */
inline __attribute__((always_inline)) constexpr Instr encodable_in(const Control control)
{
    switch (control)
    {
        case Control::Flip:
        case Control::ClearFidelity:
            return Instr::Setrwc;
    }
    return Instr::None;
}

/** @brief Instruction a field with these capabilities forces to be issued, or Instr::None when it fits more than one. */
inline __attribute__((always_inline)) constexpr Instr required_by(const Instr encodable)
{
    return encodable == Instr::Setrwc || encodable == Instr::Incrwc ? encodable : Instr::None;
}

/** @brief Test whether a field with these capabilities fits more than one instruction. */
inline __attribute__((always_inline)) constexpr bool is_flexible(const Instr encodable)
{
    return encodable == (Instr::Setrwc | Instr::Incrwc);
}

/**
 * @brief Select the instructions a descriptor issues.
 *
 * 1. Fields require SETRWC and INCRWC: issue both; flexible fields join SETRWC (see owner()).
 * 2. Fields require one instruction: issue it; flexible fields join it.
 * 3. Only flexible fields: issue INCRWC.
 * 4. Nothing to update: issue nothing; the descriptor is invalid.
 *
 * @return The issued instructions, issued in the order SETRWC, INCRWC.
 */
inline __attribute__((always_inline)) constexpr Instr plan(const MathCounter& descriptor)
{
    const Instr flip     = descriptor.flip != Src::None ? encodable_in(Control::Flip) : Instr::None;
    const Instr fidelity = descriptor.clear_fidelity ? encodable_in(Control::ClearFidelity) : Instr::None;
    const Instr src_a    = encodable_in(descriptor.src_a.op);
    const Instr src_b    = encodable_in(descriptor.src_b.op);
    const Instr dest     = encodable_in(descriptor.dest.op);

    const Instr required = required_by(flip) | required_by(fidelity) | required_by(src_a) | required_by(src_b) | required_by(dest);
    if (required != Instr::None)
    {
        return required;
    }
    if (is_flexible(flip) || is_flexible(fidelity) || is_flexible(src_a) || is_flexible(src_b) || is_flexible(dest))
    {
        return Instr::Incrwc;
    }
    return Instr::None;
}

/**
 * @brief Select the one issued instruction that encodes an operation.
 *
 * @param op: Counter operation.
 * @param issued: Instructions from plan().
 * @return SETRWC when issued and able to encode op, otherwise INCRWC when so, otherwise Instr::None.
 */
inline __attribute__((always_inline)) constexpr Instr owner(const Op op, const Instr issued)
{
    const Instr candidates = encodable_in(op) & issued;
    return issues(candidates, Instr::Setrwc) ? Instr::Setrwc : candidates;
}

/**
 * @brief Encode one counter's share of an instruction.
 *
 * @param update: Counter update.
 * @param instr: Instruction being encoded.
 * @param issued: Instructions from plan().
 * @param write_bit: Counter's bit in the SETRWC write mask.
 * @param cr_bit: Counter's bit in the CR mask.
 * @param select: Accumulates the SETRWC write mask.
 * @param carry: Accumulates the CR mask.
 * @return The counter's immediate, or zero if another instruction encodes it.
 */
inline __attribute__((always_inline)) constexpr std::uint32_t encode_counter(
    const SrcUpdate& update, const Instr instr, const Instr issued, const WriteBit write_bit, const CrBit cr_bit, std::uint32_t& select, std::uint32_t& carry)
{
    if (owner(update.op, issued) != instr)
    {
        return 0;
    }
    require(update.value <= RwcValueMax);
    select |= hal::to_underlying(write_bit);
    if (update.op == Op::AdvanceAndReset)
    {
        carry |= hal::to_underlying(cr_bit);
    }
    else if (update.op == Op::IncrementAndSave)
    {
        carry |= hal::to_underlying(CrBit::DestCToCr);
    }
    return update.value;
}

/** @brief Encode the SETRWC share of an update. */
inline __attribute__((always_inline)) constexpr std::uint32_t setrwc(const MathCounter& descriptor, const Instr issued)
{
    std::uint32_t select = 0;
    std::uint32_t carry  = 0;
    const auto dest      = encode_counter(descriptor.dest, Instr::Setrwc, issued, WriteBit::Dest, CrBit::Dest, select, carry);
    const auto src_b     = encode_counter(descriptor.src_b, Instr::Setrwc, issued, WriteBit::SrcB, CrBit::SrcB, select, carry);
    const auto src_a     = encode_counter(descriptor.src_a, Instr::Setrwc, issued, WriteBit::SrcA, CrBit::SrcA, select, carry);
    if (descriptor.clear_fidelity)
    {
        select |= hal::to_underlying(WriteBit::Fidelity);
    }
    return TT_OP_SETRWC(hal::to_underlying(descriptor.flip), carry, dest, src_b, src_a, select);
}

/** @brief Encode the INCRWC share of an update. */
inline __attribute__((always_inline)) constexpr std::uint32_t incrwc(const MathCounter& descriptor, const Instr issued)
{
    // INCRWC has no write mask: omitted counters encode a zero increment without CR.
    std::uint32_t select = 0;
    std::uint32_t carry  = 0;
    const auto dest      = encode_counter(descriptor.dest, Instr::Incrwc, issued, WriteBit::Dest, CrBit::Dest, select, carry);
    const auto src_b     = encode_counter(descriptor.src_b, Instr::Incrwc, issued, WriteBit::SrcB, CrBit::SrcB, select, carry);
    const auto src_a     = encode_counter(descriptor.src_a, Instr::Incrwc, issued, WriteBit::SrcA, CrBit::SrcA, select, carry);
    return TT_OP_INCRWC(carry, dest, src_b, src_a);
}

} // namespace detail

inline constexpr __attribute__((always_inline)) std::uint32_t MathCounter::operation() const
{
    const detail::Instr issued = detail::plan(*this);
    if (issued == (detail::Instr::Setrwc | detail::Instr::Incrwc))
    {
        detail::reject_two_instructions();
    }
    detail::require(issued != detail::Instr::None);
    return issued == detail::Instr::Setrwc ? detail::setrwc(*this, issued) : detail::incrwc(*this, issued);
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
    constexpr detail::Instr issued = detail::plan(Descriptor);
    static_assert(issued != detail::Instr::None, "math-counter descriptor must update a counter, flip a bank, or clear the fidelity phase");
    if constexpr (detail::issues(issued, detail::Instr::Setrwc))
    {
        constexpr std::uint32_t word = detail::setrwc(Descriptor, issued);
        TTI_INSN(word);
    }
    if constexpr (detail::issues(issued, detail::Instr::Incrwc))
    {
        constexpr std::uint32_t word = detail::incrwc(Descriptor, issued);
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
    const detail::Instr issued = detail::plan(descriptor);
    detail::require(issued != detail::Instr::None);
    if (detail::issues(issued, detail::Instr::Setrwc))
    {
        TT_INSN(detail::setrwc(descriptor, issued));
    }
    if (detail::issues(issued, detail::Instr::Incrwc))
    {
        TT_INSN(detail::incrwc(descriptor, issued));
    }
}

} // namespace hal::counters::rwc
