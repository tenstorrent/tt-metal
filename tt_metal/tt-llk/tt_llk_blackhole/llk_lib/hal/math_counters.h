// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

namespace hal::rwc
{

/** @brief Select math-thread register-word counters for a grouped update. */
enum class Counters : std::uint8_t
{
    None             = 0x0,
    SrcA             = 0x1,
    SrcB             = 0x2,
    Dest             = 0x4,
    Bias             = 0x8,
    All              = 0x7,
    AllIncludingBias = 0xf
};

/** @brief Select whether SETRWC preserves or clears the fidelity phase. */
enum class FidelityPhase : std::uint8_t
{
    Keep,
    Clear
};

inline constexpr Counters operator|(const Counters lhs, const Counters rhs)
{
    return static_cast<Counters>(static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs));
}

inline constexpr Counters operator&(const Counters lhs, const Counters rhs)
{
    return static_cast<Counters>(static_cast<std::uint8_t>(lhs) & static_cast<std::uint8_t>(rhs));
}

namespace detail
{

enum class Operation : std::uint8_t
{
    Unset,
    Set,
    Increment
};

constexpr std::uint8_t counter_bits(const Counters counters)
{
    return static_cast<std::uint8_t>(counters);
}

} // namespace detail

/**
 * @brief Build one grouped update to the current math thread's register-word counters.
 *
 * Each set() or increment() entry assigns one value to every counter in its mask. Disjoint
 * entries can assign different values and are folded into one SETRWC or INCRWC operation.
 * Selecting Bias adds the legacy SETIBRWC family; apply() can issue both families, while
 * get_operation() accepts only descriptions that encode as one instruction word.
 *
 * @tparam Mode: Selected update family, values = <Unset/Set/Increment>.
 * @tparam AssignedMask: Counters already assigned by set(), increment(), or advance_dest_and_save_to_carry().
 * @tparam SrcAValue: SrcA SETRWC/INCRWC immediate.
 * @tparam SrcBValue: SrcB SETRWC/INCRWC immediate.
 * @tparam DestValue: Dest SETRWC/INCRWC immediate.
 * @tparam BiasValue: Bias SETIBRWC immediate.
 * @tparam CarryBits: Carry shadows selected to advance and reload their live counters.
 * @tparam ReleaseBits: SrcA/SrcB banks released by SETRWC.
 * @tparam Fidelity: Fidelity-phase action for SETRWC.
 * @tparam DestAdvanceAndSaveToCarry: Select SETRWC's Dest counter-to-carry mode.
 */
template <
    detail::Operation Mode         = detail::Operation::Unset,
    std::uint8_t AssignedMask      = 0,
    std::uint32_t SrcAValue        = 0,
    std::uint32_t SrcBValue        = 0,
    std::uint32_t DestValue        = 0,
    std::uint32_t BiasValue        = 0,
    std::uint8_t CarryBits         = 0,
    std::uint8_t ReleaseBits       = 0,
    FidelityPhase Fidelity         = FidelityPhase::Keep,
    bool DestAdvanceAndSaveToCarry = false>
struct MathCounters
{
private:
    static constexpr std::uint8_t NORMAL_MASK = detail::counter_bits(Counters::All);
    static constexpr std::uint8_t BIAS_MASK   = detail::counter_bits(Counters::Bias);

    template <
        detail::Operation NewMode,
        std::uint8_t NewAssignedMask,
        std::uint32_t NewSrcAValue,
        std::uint32_t NewSrcBValue,
        std::uint32_t NewDestValue,
        std::uint32_t NewBiasValue,
        std::uint8_t NewCarryBits,
        std::uint8_t NewReleaseBits,
        FidelityPhase NewFidelity,
        bool NewDestAdvanceAndSaveToCarry>
    static constexpr auto rebind()
    {
        return MathCounters<
            NewMode,
            NewAssignedMask,
            NewSrcAValue,
            NewSrcBValue,
            NewDestValue,
            NewBiasValue,
            NewCarryBits,
            NewReleaseBits,
            NewFidelity,
            NewDestAdvanceAndSaveToCarry> {};
    }

    static constexpr bool has_normal_operation()
    {
        return (AssignedMask & NORMAL_MASK) != 0 || ReleaseBits != 0 || Fidelity == FidelityPhase::Clear || DestAdvanceAndSaveToCarry;
    }

    static constexpr bool has_bias_operation()
    {
        return AssignedMask & BIAS_MASK;
    }

    static constexpr void assert_valid()
    {
        static_assert(
            Mode != detail::Operation::Unset, "no math-counter operation selected — call set<>(), increment<>(), release<>(), or clear_fidelity() first");
        static_assert((AssignedMask & ~detail::counter_bits(Counters::AllIncludingBias)) == 0, "math-counter selection contains unsupported bits");
        static_assert((CarryBits & ~detail::counter_bits(Counters::AllIncludingBias)) == 0, "math-counter carry advance contains unsupported bits");
        static_assert((CarryBits & ~AssignedMask) == 0, "carry advance requires a matching set/increment entry");
        static_assert((ReleaseBits & ~detail::counter_bits(Counters::SrcA | Counters::SrcB)) == 0, "SETRWC can release only SrcA and SrcB");
        static_assert(
            !DestAdvanceAndSaveToCarry || !(CarryBits & detail::counter_bits(Counters::Dest)), "Dest cannot advance from carry and save to carry together");
        static_assert((AssignedMask & detail::counter_bits(Counters::SrcA)) == 0 || SrcAValue < 16, "SrcA RWC immediate must fit four bits");
        static_assert((AssignedMask & detail::counter_bits(Counters::SrcB)) == 0 || SrcBValue < 16, "SrcB RWC immediate must fit four bits");
        static_assert((AssignedMask & detail::counter_bits(Counters::Dest)) == 0 || DestValue < 16, "Dest RWC immediate must fit four bits");
        static_assert((AssignedMask & BIAS_MASK) == 0 || BiasValue < 4096, "Bias RWC immediate must fit twelve bits");

        if constexpr (Mode == detail::Operation::Increment)
        {
            static_assert(ReleaseBits == 0, "INCRWC cannot release source banks");
            static_assert(Fidelity == FidelityPhase::Keep, "INCRWC cannot clear the fidelity phase");
            static_assert(!DestAdvanceAndSaveToCarry, "INCRWC cannot advance Dest and save it to carry");
        }
    }

    static constexpr std::uint32_t normal_operation()
    {
        constexpr std::uint32_t normal_carry = CarryBits & NORMAL_MASK;

        if constexpr (Mode == detail::Operation::Set)
        {
            constexpr std::uint32_t counter_mask = (AssignedMask & NORMAL_MASK) | (Fidelity == FidelityPhase::Clear ? ckernel::p_setrwc::SET_F : 0);
            constexpr std::uint32_t carry        = normal_carry | (DestAdvanceAndSaveToCarry ? ckernel::p_setrwc::C_TO_CR_MODE : 0);
            return TT_OP_SETRWC(ReleaseBits, carry, DestValue, SrcBValue, SrcAValue, counter_mask);
        }
        else
        {
            static_assert(Mode == detail::Operation::Increment, "invalid math-counter operation");
            return TT_OP_INCRWC(normal_carry, DestValue, SrcBValue, SrcAValue);
        }
    }

    static constexpr std::uint32_t bias_operation()
    {
        constexpr std::uint32_t bias_carry = CarryBits & BIAS_MASK ? ckernel::p_setibrwc::CR_BIAS : ckernel::p_setibrwc::CR_NONE;
        constexpr std::uint32_t bias_mode  = Mode == detail::Operation::Increment ? ckernel::p_setibrwc::INC_BIAS : ckernel::p_setibrwc::SET_BIAS;
        return TT_OP_SETIBRWC(bias_carry, BiasValue, bias_mode);
    }

public:
    /**
     * @brief Add one value assignment to this grouped SETRWC description.
     *
     * By default, SETRWC writes the value to both the live counter and its carry shadow.
     * Selecting the counter with advance_carry_and_reload() instead adds this value to the
     * carry shadow and reloads the live counter from the result.
     *
     * @tparam SelectedCounters: Nonempty, previously unassigned counter mask.
     * @tparam Value: Value assigned to every selected counter.
     */
    template <Counters SelectedCounters, std::uint32_t Value = 0>
    constexpr auto set() const
    {
        constexpr std::uint8_t selected = detail::counter_bits(SelectedCounters);
        static_assert(Mode == detail::Operation::Unset || Mode == detail::Operation::Set, "cannot mix set and increment entries");
        static_assert(selected != 0, "set entry must select at least one counter");
        static_assert((selected & ~detail::counter_bits(Counters::AllIncludingBias)) == 0, "set entry contains unsupported counter bits");
        static_assert((AssignedMask & selected) == 0, "counter already assigned by an earlier entry");
        static_assert((selected & NORMAL_MASK) == 0 || Value < 16, "SrcA/SrcB/Dest SETRWC immediates must fit four bits");
        static_assert((selected & BIAS_MASK) == 0 || Value < 4096, "Bias SETIBRWC immediate must fit twelve bits");

        return rebind<
            detail::Operation::Set,
            AssignedMask | selected,
            selected & detail::counter_bits(Counters::SrcA) ? Value : SrcAValue,
            selected & detail::counter_bits(Counters::SrcB) ? Value : SrcBValue,
            selected & detail::counter_bits(Counters::Dest) ? Value : DestValue,
            selected & BIAS_MASK ? Value : BiasValue,
            CarryBits,
            ReleaseBits,
            Fidelity,
            DestAdvanceAndSaveToCarry>();
    }

    /**
     * @brief Add one increment assignment to this grouped INCRWC description.
     *
     * By default, INCRWC advances only the live counter. Selecting the counter with
     * advance_carry_and_reload() instead advances its carry shadow and reloads the live
     * counter from the result.
     *
     * @tparam SelectedCounters: Nonempty, previously unassigned counter mask.
     * @tparam Value: Increment applied to every selected counter.
     */
    template <Counters SelectedCounters, std::uint32_t Value = 0>
    constexpr auto increment() const
    {
        constexpr std::uint8_t selected = detail::counter_bits(SelectedCounters);
        static_assert(Mode == detail::Operation::Unset || Mode == detail::Operation::Increment, "cannot mix set and increment entries");
        static_assert(selected != 0, "increment entry must select at least one counter");
        static_assert((selected & ~detail::counter_bits(Counters::AllIncludingBias)) == 0, "increment entry contains unsupported counter bits");
        static_assert((AssignedMask & selected) == 0, "counter already assigned by an earlier entry");
        static_assert((selected & NORMAL_MASK) == 0 || Value < 16, "SrcA/SrcB/Dest INCRWC immediates must fit four bits");
        static_assert((selected & BIAS_MASK) == 0 || Value < 4096, "Bias SETIBRWC immediate must fit twelve bits");

        return rebind<
            detail::Operation::Increment,
            AssignedMask | selected,
            selected & detail::counter_bits(Counters::SrcA) ? Value : SrcAValue,
            selected & detail::counter_bits(Counters::SrcB) ? Value : SrcBValue,
            selected & detail::counter_bits(Counters::Dest) ? Value : DestValue,
            selected & BIAS_MASK ? Value : BiasValue,
            CarryBits,
            ReleaseBits,
            Fidelity,
            DestAdvanceAndSaveToCarry>();
    }

    /**
     * @brief Advance selected carry shadows, then reload their live counters.
     *
     * Each selected shadow advances by the immediate from its matching set() or increment()
     * entry. The updated shadow is then copied into the corresponding live counter.
     *
     * @tparam SelectedCounters: Counters whose carry shadows advance and reload the live value.
     */
    template <Counters SelectedCounters>
    constexpr auto advance_carry_and_reload() const
    {
        constexpr std::uint8_t selected = detail::counter_bits(SelectedCounters);
        static_assert(selected != 0, "carry advance must select at least one counter");
        static_assert((selected & ~detail::counter_bits(Counters::AllIncludingBias)) == 0, "carry advance contains unsupported counter bits");
        static_assert((CarryBits & selected) == 0, "counter carry shadow already selected for advance and reload");

        return rebind<Mode, AssignedMask, SrcAValue, SrcBValue, DestValue, BiasValue, CarryBits | selected, ReleaseBits, Fidelity, DestAdvanceAndSaveToCarry>();
    }

    /**
     * @brief Complete selected source-bank handoffs and switch banks while issuing SETRWC.
     *
     * SETRWC switches Math from each selected current source bank to its paired bank. Normally it
     * also clears the consumed bank's data-valid state, returning that bank to its unpacker. The
     * per-thread CLR_DVALID disable configuration preserves data-valid while still switching banks.
     *
     * @tparam Sources: SrcA, SrcB, or both.
     */
    template <Counters Sources>
    constexpr auto release() const
    {
        constexpr std::uint8_t selected = detail::counter_bits(Sources);
        static_assert(Mode == detail::Operation::Unset || Mode == detail::Operation::Set, "INCRWC cannot release source banks");
        static_assert(selected != 0, "release must select SrcA, SrcB, or both");
        static_assert((selected & ~detail::counter_bits(Counters::SrcA | Counters::SrcB)) == 0, "release can select only SrcA and SrcB");
        static_assert((ReleaseBits & selected) == 0, "source bank already selected for release");

        return rebind<
            detail::Operation::Set,
            AssignedMask,
            SrcAValue,
            SrcBValue,
            DestValue,
            BiasValue,
            CarryBits,
            ReleaseBits | selected,
            Fidelity,
            DestAdvanceAndSaveToCarry>();
    }

    /** @brief Clear the fidelity phase while issuing SETRWC. */
    constexpr auto clear_fidelity() const
    {
        static_assert(Mode == detail::Operation::Unset || Mode == detail::Operation::Set, "INCRWC cannot clear the fidelity phase");
        static_assert(Fidelity == FidelityPhase::Keep, "fidelity phase already selected for clearing");

        return rebind<
            detail::Operation::Set,
            AssignedMask,
            SrcAValue,
            SrcBValue,
            DestValue,
            BiasValue,
            CarryBits,
            ReleaseBits,
            FidelityPhase::Clear,
            DestAdvanceAndSaveToCarry>();
    }

    /**
     * @brief Advance Dest by an immediate and save the result into its carry shadow.
     *
     * @tparam Value: Four-bit Dest increment; zero saves the current Dest counter to its shadow.
     */
    template <std::uint32_t Value = 0>
    constexpr auto advance_dest_and_save_to_carry() const
    {
        constexpr std::uint8_t dest = detail::counter_bits(Counters::Dest);
        static_assert(Mode == detail::Operation::Unset || Mode == detail::Operation::Set, "advancing Dest and saving to carry belongs to SETRWC");
        static_assert(!(AssignedMask & dest), "Dest already assigned by an earlier entry");
        static_assert(Value < 16, "Dest SETRWC immediate must fit four bits");
        static_assert(!DestAdvanceAndSaveToCarry, "Dest already selected to advance and save to carry");

        return rebind<detail::Operation::Set, AssignedMask | dest, SrcAValue, SrcBValue, Value, BiasValue, CarryBits, ReleaseBits, Fidelity, true>();
    }

    /**
     * @brief Encode a single-instruction math-counter description without issuing it.
     *
     * @note Use @ref apply when the description combines Bias with SrcA/SrcB/Dest,
     *       source release, fidelity clear, or advancing Dest and saving it to carry.
     */
    static constexpr std::uint32_t get_operation()
    {
        assert_valid();
        static_assert(
            has_normal_operation() != has_bias_operation(),
            "get_operation() requires exactly one instruction; use apply() when Bias is combined with SETRWC/INCRWC state");

        if constexpr (has_bias_operation())
        {
            return bias_operation();
        }
        else
        {
            return normal_operation();
        }
    }

    /**
     * @brief Issue the grouped math-counter description.
     *
     * Emits one SETRWC or INCRWC for SrcA/SrcB/Dest state and one SETIBRWC when Bias is
     * selected. A Bias-only or normal-only description emits exactly one instruction.
     */
    inline __attribute__((always_inline)) void apply() const
    {
        assert_valid();

        if constexpr (has_normal_operation())
        {
            INSTRUCTION_WORD(normal_operation());
        }
        if constexpr (has_bias_operation())
        {
            INSTRUCTION_WORD(bias_operation());
        }
    }
};

} // namespace hal::rwc

namespace hal
{

inline constexpr rwc::MathCounters<> math_counters {};

} // namespace hal
