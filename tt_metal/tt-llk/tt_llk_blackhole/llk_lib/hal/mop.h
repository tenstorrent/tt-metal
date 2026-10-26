// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "llk_assert.h"

namespace hal::mop
{

/**
 * @brief Select a MOP Expander template.
 */
enum class MopTemplate : std::uint8_t
{
    Template0,
    Template1
};

namespace detail
{
inline constexpr std::uint32_t COUNT_WIDTH = 10;

constexpr bool is_nop(const std::uint32_t instruction)
{
    return (instruction >> 24) == (TT_OP_NOP >> 24);
}

constexpr bool is_valid_count(const std::uint32_t count)
{
    return ckernel::is_valid(count, COUNT_WIDTH);
}

// The Blackhole expander adds to OuterCount when the outer loop has no start op, the inner loop is
// empty, and an end op exists: +1024 if EndOp1 is set and OuterCount is 1, +1 for any OuterCount otherwise.
constexpr bool triggers_count_anomaly(
    const std::uint32_t outer_count,
    const std::uint32_t inner_count,
    const std::uint32_t start_op,
    const std::uint32_t end_op,
    const std::uint32_t second_end_op)
{
    return is_nop(start_op) && inner_count == 0 && !is_nop(end_op) && (is_nop(second_end_op) || outer_count == 1);
}

inline __attribute__((always_inline)) void assert_valid_count(const bool valid)
{
    LLK_ASSERT(valid, "Template 1 MOP loop count must fit in 10 bits");
}

inline __attribute__((always_inline)) void assert_no_count_anomaly(const bool valid)
{
    LLK_ASSERT(valid, "Template 1 MOP configuration triggers the no-start/zero-inner/end-op outer count anomaly");
}

inline __attribute__((always_inline)) void assert_patch_reachable(const bool valid)
{
    LLK_ASSERT(valid, "Template 1 MOP last-iteration ops are never issued with an empty inner loop");
}

inline __attribute__((always_inline)) void assert_valid_run_count(const bool valid)
{
    LLK_ASSERT(valid, "MOP encoded count must fit in 7 bits");
}

inline __attribute__((always_inline)) void assert_mask_high_unused(const bool valid)
{
    LLK_ASSERT(valid, "Template 0 MOP operation cannot write the high mask bits; write them with configure_mask_high and set mask_high to MaskHigh::Preset");
}

// Fails constant evaluation of a constexpr configuration, or reports through Assert when built at runtime.
template <void (*Assert)(bool)>
constexpr void check(const bool valid)
{
    if (__builtin_is_constant_evaluated())
    {
        if (!valid)
        {
            __builtin_trap();
        }
    }
    else
    {
        Assert(valid);
    }
}

template <bool HasA123, bool HasB>
struct Template0Plan
{
};

template <bool HasEndOp>
struct Template1Plan
{
};
} // namespace detail

/**
 * @brief Hold a complete MOP Expander configuration built by @ref foreach_mask_bit or @ref loop.
 *
 * @tparam Plan: Compile-time shape that selects which MopCfg words @ref program writes.
 * @note Spell the type as `mop::cfg`; class template argument deduction recovers the plan.
 */
template <typename Plan>
struct cfg;

// ---------------- Template 0 ----------------

/**
 * @brief Hold the ops issued for a clear mask bit.
 *
 * @tparam HasA123: Whether InsnA1-3 are issued after InsnA0.
 * @tparam HasB: Whether InsnB is issued last.
 */
template <bool HasA123, bool HasB>
struct IfClear
{
    std::uint32_t a0;
    std::uint32_t a1;
    std::uint32_t a2;
    std::uint32_t a3;
    std::uint32_t b;
};

/**
 * @brief Hold the ops issued for a set mask bit.
 *
 * @tparam HasSkipB: Whether SkipB is issued after SkipA0.
 */
template <bool HasSkipB>
struct Otherwise
{
    std::uint32_t skip_a0;
    std::uint32_t skip_b;
};

/**
 * @brief Describe the ops issued, in order, for a clear mask bit.
 *
 * The instruction count selects the shape: 1 = {a0}, 2 = {a0, b}, 4 = {a0, a1, a2, a3},
 * 5 = {a0, a1, a2, a3, b}.
 *
 * @param a0: First op of every clear-bit iteration (InsnA0).
 * @param rest: Remaining encoded instructions, InsnA1-3 and/or InsnB.
 * @note The expander issues InsnA1-3 as a group, NOPs included.
 */
template <typename... Instructions>
constexpr auto if_clear(const std::uint32_t a0, const Instructions... rest)
{
    constexpr std::uint32_t count = 1 + sizeof...(Instructions);
    static_assert(count == 1 || count == 2 || count == 4 || count == 5, "Template 0 MOP if_clear takes {a0}, {a0, b}, {a0, a1, a2, a3} or {a0, a1, a2, a3, b}");
    static_assert((std::is_convertible_v<Instructions, std::uint32_t> && ...), "Template 0 MOP if_clear takes encoded instructions");

    constexpr bool has_a123            = count >= 4;
    constexpr bool has_b               = count == 2 || count == 5;
    const std::uint32_t instructions[] = {a0, static_cast<std::uint32_t>(rest)...};

    IfClear<has_a123, has_b> clear = {a0, TT_OP_NOP, TT_OP_NOP, TT_OP_NOP, TT_OP_NOP};
    if constexpr (has_a123)
    {
        clear.a1 = instructions[1];
        clear.a2 = instructions[2];
        clear.a3 = instructions[3];
    }
    if constexpr (has_b)
    {
        clear.b = instructions[count - 1];
    }
    return clear;
}

/** @brief Issue skip_a0 on a set mask bit. */
constexpr Otherwise<false> otherwise(const std::uint32_t skip_a0)
{
    return {skip_a0, TT_OP_NOP};
}

/**
 * @brief Issue skip_a0, then skip_b, on a set mask bit.
 *
 * @note Pair with an @ref if_clear that has a b op; the expander issues both or neither.
 */
constexpr Otherwise<true> otherwise(const std::uint32_t skip_a0, const std::uint32_t skip_b)
{
    return {skip_a0, skip_b};
}

template <bool HasA123, bool HasB>
struct cfg<detail::Template0Plan<HasA123, HasB>>
{
    std::uint32_t a0;
    std::uint32_t a1;
    std::uint32_t a2;
    std::uint32_t a3;
    std::uint32_t b;
    std::uint32_t skip_a0;
    std::uint32_t skip_b;
};

/**
 * @brief Build a Template 0 configuration that issues only clear-bit ops.
 *
 * @param clear: Ops issued for each clear mask bit.
 * @note A set mask bit issues NOPs; run with a mask whose iterated bits are clear.
 */
template <bool HasA123, bool HasB>
constexpr cfg<detail::Template0Plan<HasA123, HasB>> foreach_mask_bit(const IfClear<HasA123, HasB> clear)
{
    return {clear.a0, clear.a1, clear.a2, clear.a3, clear.b, TT_OP_NOP, TT_OP_NOP};
}

/**
 * @brief Build a Template 0 configuration that selects ops per mask bit.
 *
 * @param clear: Ops issued for each clear mask bit.
 * @param set: Ops issued for each set mask bit.
 */
template <bool HasA123, bool HasB, bool HasSkipB>
constexpr cfg<detail::Template0Plan<HasA123, HasB>> foreach_mask_bit(const IfClear<HasA123, HasB> clear, const Otherwise<HasSkipB> set)
{
    static_assert(HasB == HasSkipB, "Template 0 MOP issues InsnB and SkipB under one flag; give if_clear a b op exactly when otherwise has a skip_b op");

    return {clear.a0, clear.a1, clear.a2, clear.a3, clear.b, set.skip_a0, set.skip_b};
}

// ---------------- Template 1 ----------------

/**
 * @brief Hold the inner loop of a Template 1 configuration.
 */
struct InnerLoop
{
    std::uint32_t count;
    std::uint32_t op;           // LoopOp
    std::uint32_t alternate_op; // LoopOp1
};

/**
 * @brief Hold one Template 1 MopCfg word written by @ref update.
 *
 * @tparam Word: MopCfg word, values = <0 (outer count)/1 (inner count)/7 (last outer op)/8 (last inner op)>.
 */
template <std::uint32_t Word>
struct Update
{
    std::uint32_t value;
};

/**
 * @brief Name one Template 1 MopCfg word; call it with a value to build an @ref Update.
 *
 * @tparam Word: MopCfg word, values = <0 (outer count)/1 (inner count)/7 (last outer op)/8 (last inner op)>.
 */
template <std::uint32_t Word>
struct Field
{
    constexpr Update<Word> operator()(const std::uint32_t value) const
    {
        return {value};
    }
};

inline constexpr Field<0> outer_count {};
inline constexpr Field<1> inner_count {};
inline constexpr Field<7> last_outer {}; // Loop0Last: last inner iteration of the last outer iteration
inline constexpr Field<8> last_inner {}; // Loop1Last: last inner iteration of every other outer iteration

template <bool HasEndOp>
struct cfg<detail::Template1Plan<HasEndOp>>
{
    std::uint32_t outer_count;
    std::uint32_t inner_count;
    std::uint32_t start_op;
    std::uint32_t end_op;
    std::uint32_t second_end_op;
    std::uint32_t op;
    std::uint32_t alternate_op;
    std::uint32_t last_outer_op;
    std::uint32_t last_inner_op;

    /**
     * @brief Return a copy that issues instruction in place of the selected last-iteration op.
     *
     * @param last: Iteration whose final inner op is replaced, values = <last_outer/last_inner>.
     * @param instruction: Encoded instruction issued instead.
     * @note Without a patch, both last-iteration ops repeat the last op of the inner loop body.
     */
    template <std::uint32_t Word>
    [[nodiscard]] constexpr cfg patch(const Field<Word> last, const std::uint32_t instruction) const
    {
        static_assert(Word == 7 || Word == 8, "Template 1 MOP patch replaces last_outer or last_inner");
        detail::check<detail::assert_patch_reachable>(inner_count != 0);

        cfg patched = *this;
        if constexpr (Word == 7)
        {
            patched.last_outer_op = last(instruction).value;
        }
        else
        {
            patched.last_inner_op = last(instruction).value;
        }
        return patched;
    }
};

/**
 * @brief Build a Template 1 inner loop that issues op count times.
 *
 * @param count: Inner iteration count, valid values = <0-1023>.
 * @param op: Op issued every iteration (LoopOp).
 */
constexpr InnerLoop loop(const std::uint32_t count, const std::uint32_t op)
{
    return {count, op, TT_OP_NOP};
}

/**
 * @brief Build a Template 1 inner loop that issues op, then alternate_op, count times.
 *
 * @param count: Inner iteration count, valid values = <0-1023>.
 * @param op: First op of every iteration (LoopOp).
 * @param alternate_op: Second op of every iteration (LoopOp1); a NOP disables it.
 */
constexpr InnerLoop loop(const std::uint32_t count, const std::uint32_t op, const std::uint32_t alternate_op)
{
    return {count, op, alternate_op};
}

namespace detail
{
template <bool HasEndOp>
constexpr cfg<Template1Plan<HasEndOp>> make_loop(
    const std::uint32_t outer_count, const std::uint32_t start_op, const InnerLoop inner, const std::uint32_t end_op, const std::uint32_t second_end_op)
{
    check<assert_valid_count>(is_valid_count(outer_count));
    check<assert_valid_count>(is_valid_count(inner.count));
    check<assert_no_count_anomaly>(!triggers_count_anomaly(outer_count, inner.count, start_op, end_op, second_end_op));

    // The expander replaces the last op of the body, so repeating it keeps the issued sequence unchanged.
    const std::uint32_t last_op = is_nop(inner.alternate_op) ? inner.op : inner.alternate_op;

    return {outer_count, inner.count, start_op, end_op, second_end_op, inner.op, inner.alternate_op, last_op, last_op};
}
} // namespace detail

/**
 * @brief Build a Template 1 configuration around an inner loop.
 *
 * @param count: Outer iteration count, valid values = <0-1023>.
 * @param inner: Inner loop built by the two- or three-argument @ref loop.
 * @note Overloads add a start op before the inner loop and up to two end ops after it.
 */
constexpr cfg<detail::Template1Plan<false>> loop(const std::uint32_t count, const InnerLoop inner)
{
    return detail::make_loop<false>(count, TT_OP_NOP, inner, TT_OP_NOP, TT_OP_NOP);
}

/** @brief Build a Template 1 configuration that issues start_op before each inner loop. */
constexpr cfg<detail::Template1Plan<false>> loop(const std::uint32_t count, const std::uint32_t start_op, const InnerLoop inner)
{
    return detail::make_loop<false>(count, start_op, inner, TT_OP_NOP, TT_OP_NOP);
}

/** @brief Build a Template 1 configuration that issues end_op after each inner loop. */
constexpr cfg<detail::Template1Plan<true>> loop(const std::uint32_t count, const InnerLoop inner, const std::uint32_t end_op)
{
    return detail::make_loop<true>(count, TT_OP_NOP, inner, end_op, TT_OP_NOP);
}

/** @brief Build a Template 1 configuration that issues end_op, then second_end_op, after each inner loop. */
constexpr cfg<detail::Template1Plan<true>> loop(const std::uint32_t count, const InnerLoop inner, const std::uint32_t end_op, const std::uint32_t second_end_op)
{
    return detail::make_loop<true>(count, TT_OP_NOP, inner, end_op, second_end_op);
}

/** @brief Build a Template 1 configuration that issues start_op before and end_op after each inner loop. */
constexpr cfg<detail::Template1Plan<true>> loop(const std::uint32_t count, const std::uint32_t start_op, const InnerLoop inner, const std::uint32_t end_op)
{
    return detail::make_loop<true>(count, start_op, inner, end_op, TT_OP_NOP);
}

/** @brief Build a Template 1 configuration that issues start_op before and two end ops after each inner loop. */
constexpr cfg<detail::Template1Plan<true>> loop(
    const std::uint32_t count, const std::uint32_t start_op, const InnerLoop inner, const std::uint32_t end_op, const std::uint32_t second_end_op)
{
    return detail::make_loop<true>(count, start_op, inner, end_op, second_end_op);
}

// ---------------- Programming ----------------

/**
 * @brief Program this thread's MopCfg with a Template 0 configuration.
 *
 * Writes the flags, InsnA0 and SkipA0 always; InsnA1-3 only when @ref if_clear has them, and
 * InsnB/SkipB only with a b op. Skipped words keep their previous values, which the
 * expander does not read.
 *
 * @param config: Configuration built by @ref foreach_mask_bit.
 * @note Waits for the preceding MOP expansion before writing. Run it with
 *       @ref Runner<MopTemplate::Template0>.
 */
template <bool HasA123, bool HasB>
inline __attribute__((always_inline)) void program(const cfg<detail::Template0Plan<HasA123, HasB>> &config)
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);
    constexpr std::uint32_t flags   = static_cast<std::uint32_t>(HasB) | (static_cast<std::uint32_t>(HasA123) << 1);

    // Copy the configuration into scalars before the sync; reading the aggregate between the
    // volatile MopCfg accesses forces it onto the stack instead of into registers.
    const std::uint32_t b       = config.b;
    const std::uint32_t a0      = config.a0;
    const std::uint32_t a1      = config.a1;
    const std::uint32_t a2      = config.a2;
    const std::uint32_t a3      = config.a3;
    const std::uint32_t skip_a0 = config.skip_a0;
    const std::uint32_t skip_b  = config.skip_b;

    ckernel::mop_sync();

    mop_cfg[1] = flags;
    if constexpr (HasB)
    {
        mop_cfg[2] = b;
    }
    mop_cfg[3] = a0;
    if constexpr (HasA123)
    {
        mop_cfg[4] = a1;
        mop_cfg[5] = a2;
        mop_cfg[6] = a3;
    }
    mop_cfg[7] = skip_a0;
    if constexpr (HasB)
    {
        mop_cfg[8] = skip_b;
    }
}

/**
 * @brief Program this thread's MopCfg with a Template 1 configuration.
 *
 * Writes the counts, StartOp and EndOp0 always; EndOp1 only when the configuration has end
 * ops, and the inner-loop and last-iteration ops only when the inner count is nonzero.
 * Skipped words keep their previous values, which the expander does not read.
 *
 * @param config: Configuration built by @ref loop, optionally patched with @ref cfg::patch.
 * @note Waits for the preceding MOP expansion before writing. Run it with
 *       @ref Runner<MopTemplate::Template1>.
 */
template <bool HasEndOp>
inline __attribute__((always_inline)) void program(const cfg<detail::Template1Plan<HasEndOp>> &config)
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    // Copy the configuration into scalars before the sync; reading the aggregate between the
    // volatile MopCfg accesses forces it onto the stack instead of into registers.
    const std::uint32_t outer_count   = config.outer_count;
    const std::uint32_t inner_count   = config.inner_count;
    const std::uint32_t start_op      = config.start_op;
    const std::uint32_t end_op        = config.end_op;
    const std::uint32_t second_end_op = config.second_end_op;
    const std::uint32_t op            = config.op;
    const std::uint32_t alternate_op  = config.alternate_op;
    const std::uint32_t last_outer_op = config.last_outer_op;
    const std::uint32_t last_inner_op = config.last_inner_op;

    ckernel::mop_sync();

    mop_cfg[0] = outer_count;
    mop_cfg[1] = inner_count;
    mop_cfg[2] = start_op;
    mop_cfg[3] = end_op;
    if constexpr (HasEndOp)
    {
        mop_cfg[4] = second_end_op;
    }
    if (inner_count != 0)
    {
        mop_cfg[5] = op;
        mop_cfg[6] = alternate_op;
        mop_cfg[7] = last_outer_op;
        mop_cfg[8] = last_inner_op;
    }
}

// ---------------- Updates ----------------

namespace detail
{
template <std::uint32_t Word, std::uint32_t... Words>
inline constexpr std::uint32_t occurrences = (0u + ... + static_cast<std::uint32_t>(Word == Words));
} // namespace detail

/**
 * @brief Replace Template 1 MopCfg words in this thread's MopCfg, waiting for the preceding expansion once.
 *
 * @param updates: Distinct words built with @ref outer_count, @ref inner_count, @ref last_outer or @ref last_inner.
 * @note Program a Template 1 configuration with @ref program first. MopCfg is write-only, so
 *       the new words are not checked against it: keep the inner count nonzero when the
 *       programmed one was nonzero (and vice versa, since an empty inner loop leaves its op
 *       words unwritten), and keep the result clear of the count anomaly rejected by @ref loop.
 */
template <std::uint32_t... Words>
inline __attribute__((always_inline)) void update(const Update<Words>... updates)
{
    static_assert(sizeof...(Words) >= 1, "Template 1 MOP update takes at least one word");
    static_assert(((detail::occurrences<Words, Words...> == 1) && ...), "Template 1 MOP update takes each word at most once");

    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    (detail::assert_valid_count(Words > 1 || detail::is_valid_count(updates.value)), ...);

    ckernel::mop_sync();

    ((mop_cfg[Words] = updates.value), ...);
}

/**
 * @brief Replace one Template 1 MopCfg word in this thread's MopCfg.
 *
 * @param field: Word to replace, values = <outer_count/inner_count/last_outer/last_inner>.
 * @param value: New count (valid values = <0-1023>) or encoded instruction.
 * @note See the variadic overload for the caller's obligations.
 */
template <std::uint32_t Word>
inline __attribute__((always_inline)) void update(const Field<Word> field, const std::uint32_t value)
{
    update(field(value));
}

// ---------------- Running ----------------

/**
 * @brief Select who writes the high 16 bits of a Template 0 mask, which runs of more than 16 iterations read.
 */
enum class MaskHigh : std::uint8_t
{
    Auto,   // run writes them with MOP_CFG when it may read them; operation rejects a run that reads them
    Preset, // the caller wrote them with configure_mask_high; nothing writes or checks them
};

/**
 * @brief Issue a MOP expansion for the selected template.
 *
 * @tparam tmpl: MOP template to run, values = <Template0/Template1>.
 */
template <MopTemplate tmpl>
struct Runner;

/**
 * @brief Issue a Template 0 MOP expansion, or describe one and encode it with @ref operation.
 *
 * `Runner<MopTemplate::Template0> {.count = 3, .mask = faces}.operation()` returns the MOP instruction
 * the matching run would issue, for issuing later with TT_INSN.
 */
template <>
struct Runner<MopTemplate::Template0>
{
private:
    static constexpr std::uint8_t COUNT_WIDTH   = 7;
    static constexpr std::uint8_t MASK_LO_WIDTH = 16;
    static constexpr std::uint32_t MASK_LO      = (1u << MASK_LO_WIDTH) - 1;

    static constexpr bool is_valid_count(const std::uint32_t count)
    {
        return ckernel::is_valid(count - 1, COUNT_WIDTH);
    }

    static constexpr bool reads_mask_high(const std::uint32_t count, const std::uint32_t mask)
    {
        return count > MASK_LO_WIDTH || !ckernel::is_valid(mask, MASK_LO_WIDTH);
    }

public:
    std::uint32_t count;    // number of expansion iterations, valid values = <1-128>
    std::uint32_t mask = 0; // per-iteration selector mask, consumed least-significant bit first
    MaskHigh mask_high = MaskHigh::Auto;

    /**
     * @brief Encode this run as a MOP instruction.
     *
     * @note With @ref MaskHigh::Auto, count must be at most 16 and mask must fit in 16 bits, since the
     *       encoding cannot carry the MOP_CFG that writes the high mask bits.
     */
    [[nodiscard]] constexpr std::uint32_t operation() const
    {
        detail::check<detail::assert_valid_run_count>(is_valid_count(count));
        if (mask_high == MaskHigh::Auto)
        {
            detail::check<detail::assert_mask_high_unused>(!reads_mask_high(count, mask));
        }
        return TT_OP_MOP(0 /*mop_type*/, count - 1, mask & MASK_LO);
    }

    /**
     * @brief Write the high 16 bits of a compile-time mask for later runs with @ref MaskHigh::Preset.
     *
     * @tparam Mask: Full per-iteration selector mask; only bits 16-31 are written.
     */
    template <std::uint32_t Mask>
    inline __attribute__((always_inline)) static void configure_mask_high()
    {
        TTI_MOP_CFG(Mask >> MASK_LO_WIDTH);
    }

    /**
     * @brief Write the high 16 bits of a runtime mask for later runs with @ref MaskHigh::Preset.
     *
     * @param mask: Full per-iteration selector mask; only bits 16-31 are written.
     */
    inline __attribute__((always_inline)) static void configure_mask_high(const std::uint32_t mask)
    {
        TT_MOP_CFG(mask >> MASK_LO_WIDTH);
    }

    /**
     * @brief Run the programmed Template 0 configuration with compile-time arguments.
     *
     * @tparam Count: Number of expansion iterations, valid values = <1-128>.
     * @tparam Mask: Per-iteration selector mask, consumed least-significant bit first.
     * @tparam High: Who writes the high mask bits, values = <Auto/Preset>.
     * @note Call @ref program with a Template 0 configuration before this function.
     */
    template <std::uint32_t Count, std::uint32_t Mask, MaskHigh High = MaskHigh::Auto>
    inline __attribute__((always_inline)) static void run()
    {
        static_assert(is_valid_count(Count), "MOP encoded count must fit in 7 bits");

        if constexpr (High == MaskHigh::Auto && reads_mask_high(Count, Mask))
        {
            configure_mask_high<Mask>();
        }
        TTI_MOP(0, Count - 1, Mask & MASK_LO);
    }

    /**
     * @brief Run the programmed Template 0 configuration with runtime count.
     *
     * @tparam Mask: Compile-time per-iteration selector mask, consumed least-significant bit first.
     * @tparam High: Who writes the high mask bits, values = <Auto/Preset>.
     * @param count: Number of expansion iterations, valid values = <1-128>.
     * @note Call @ref program with a Template 0 configuration before this function.
     */
    template <std::uint32_t Mask, MaskHigh High = MaskHigh::Auto>
    inline __attribute__((always_inline)) static void run(const std::uint32_t count)
    {
        LLK_ASSERT(is_valid_count(count), "MOP encoded count must fit in 7 bits");

        if constexpr (High == MaskHigh::Auto)
        {
            configure_mask_high<Mask>();
        }
        TT_MOP(0, count - 1, Mask & MASK_LO);
    }

    /**
     * @brief Run the programmed Template 0 configuration with runtime mask.
     *
     * @tparam Count: Compile-time number of expansion iterations, valid values = <1-128>.
     * @tparam High: Who writes the high mask bits, values = <Auto/Preset>.
     * @param mask: Per-iteration selector mask, consumed least-significant bit first.
     * @note Call @ref program with a Template 0 configuration before this function.
     */
    template <std::uint32_t Count, MaskHigh High = MaskHigh::Auto>
    inline __attribute__((always_inline)) static void run_with_runtime_mask(const std::uint32_t mask)
    {
        static_assert(is_valid_count(Count), "MOP encoded count must fit in 7 bits");

        if constexpr (High == MaskHigh::Auto)
        {
            configure_mask_high(mask);
        }
        TT_MOP(0, Count - 1, mask & MASK_LO);
    }

    /**
     * @brief Run the programmed Template 0 configuration with runtime arguments.
     *
     * @tparam High: Who writes the high mask bits, values = <Auto/Preset>.
     * @param count: Number of expansion iterations, valid values = <1-128>.
     * @param mask: Per-iteration selector mask, consumed least-significant bit first.
     * @note Call @ref program with a Template 0 configuration before this function.
     */
    template <MaskHigh High = MaskHigh::Auto>
    inline __attribute__((always_inline)) static void run(const std::uint32_t count, const std::uint32_t mask)
    {
        LLK_ASSERT(is_valid_count(count), "MOP encoded count must fit in 7 bits");

        if constexpr (High == MaskHigh::Auto)
        {
            configure_mask_high(mask);
        }
        TT_MOP(0, count - 1, mask & MASK_LO);
    }
};

/**
 * @brief Issue or encode a Template 1 MOP expansion using the programmed loop counts.
 */
template <>
struct Runner<MopTemplate::Template1>
{
    /**
     * @brief Run the programmed Template 1 configuration.
     *
     * @note Call @ref program with a Template 1 configuration before this function.
     */
    inline __attribute__((always_inline)) static void run()
    {
        TTI_MOP(1, 0, 0);
    }

    /** @brief Encode @ref run as a MOP instruction. */
    [[nodiscard]] constexpr std::uint32_t operation() const
    {
        return TT_OP_MOP(1 /*mop_type*/, 0 /*loop_count*/, 0 /*zmask_lo16_or_loop_count*/);
    }
};

} // namespace hal::mop
