// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file
 * @brief Describe, encode, and issue Blackhole address-counter updates.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "../utils/gpr.h"
#include "../utils/helpers.h"
#include "ckernel.h"
#include "llk_assert.h"

/** @brief Address counters for unpackers and packers on the issuing Tensix thread. */
namespace hal::counters::adc
{

/** @brief Select address counters belonging to either unpacker or all four packers. */
enum class Client : std::uint8_t
{
    Unpacker0 = 0x1, ///< Counter set for unpacker 0.
    Unpacker1 = 0x2, ///< Counter set for unpacker 1.
    Packers   = 0x4  ///< Counter set shared by all four packers.
};

/** @brief Select the effect on each explicitly supplied counter. */
enum class Action : std::uint8_t
{
    Set,            ///< Assign the supplied value to both the live counter and its carriage-return (CR) shadow.
    Increment,      ///< Add the supplied value to the live counter; preserve its CR shadow.
    AdvanceAndReset ///< Add the supplied value to the CR shadow, then copy the updated shadow to the live counter.
};

/**
 * @brief An optional counter value. Omission preserves the counter; an explicit zero selects it.
 *
 * Public members keep descriptors usable as template arguments. Normally initialize a field
 * directly with its value, for example .x = 10. GPR sources are rejected on Blackhole.
 */
class Value
{
public:
    std::uint32_t value = 0;     ///< Assignment or increment, interpreted according to AddrCounter::action.
    bool selected       = false; ///< Whether this counter participates in the update, including when its value is zero.

    /** @brief Omit a counter from the update, preserving its live value and CR shadow. */
    constexpr Value() = default;

    /**
     * @brief Select a counter and supply its assignment or increment.
     *
     * @param value: Unsigned assignment or increment; zero still selects the counter.
     * @note Construction stores the value without checking its range. Range checks are deferred
     *       to the descriptor's encoding or issue operation.
     */
    constexpr __attribute__((always_inline)) Value(const std::uint32_t value) : value(value), selected(true)
    {
    }

    /**
     * @brief Reject GPR-sourced counter updates at compile time.
     *
     * @tparam Index: Source GPR type index.
     * @note Every instantiation fails: Blackhole REG2FLOP is unsupported because of hardware issues.
     */
    template <std::uint32_t Index>
    constexpr Value(const hal::Gpr<Index>)
    {
        static_assert(hal::always_false<Index>, "Blackhole does not handle REG2FLOP properly. GPR address-counter writes are unsupported");
    }
};

/** @brief Sparse X/Y/Z/W values for one address-counter channel. */
class Channel
{
public:
    Value x {}; ///< X assignment or increment; omitted by default.
    Value y {}; ///< Y assignment or increment; omitted by default.
    Value z {}; ///< Z assignment or increment; omitted by default.
    Value w {}; ///< W assignment or increment; omitted by default.
};

/**
 * @brief Describe an address-counter update independently of its instruction encoding.
 *
 * Each channel has its own sparse X/Y/Z/W assignments. Omitted fields and channels remain
 * unchanged. With AdvanceAndReset, an explicit zero reloads the live counter from its shadow.
 * Clients selected by for_clients() receive the same update on the issuing Tensix thread.
 *
 * X participates in input address generation and datum counting on both channels. For Y/Z/W,
 * unpacker channel0 addresses input and channel1 addresses output; packer channel0 addresses
 * Dst input and channel1 addresses L1 output. All four packers share the Packers counters.
 *
 * @note Set assigns both live counters and their CR shadows. Increment preserves the shadows.
 *       Independent assignment to a CR shadow alone and thread overrides are not supported.
 *
 * @code
 * constexpr AddrCounter matmul = AddrCounter {
 *     .channel0 = {.x = 10, .y = 20},
 *     .channel1 = {.y = 0},
 * }.for_clients<Client::Unpacker0, Client::Unpacker1>();
 * set<matmul>();
 *
 * constexpr AddrCounter reset_xy = AddrCounter {
 *     .channel0 = {.x = 0, .y = 0},
 * }.for_clients<Client::Unpacker0>();
 * constexpr auto reset_xy_word = reset_xy.operation();
 * @endcode
 */
class AddrCounter
{
public:
    Action action = Action::Set; ///< Effect applied to every selected counter.
    Channel channel0 {};         ///< Sparse assignments or increments for channel 0.
    Channel channel1 {};         ///< Sparse assignments or increments for channel 1.
    std::uint8_t clients = 0;    ///< Destination client bitmask; populate with for_clients() before encoding or issuing.

    /**
     * @brief Copy the descriptor with a new destination client set.
     *
     * @tparam Clients: One or more valid Client enumerators receiving the same update.
     * @return A copy with its client mask replaced; the original descriptor is unchanged.
     * @note An empty pack or an invalid enumerator is rejected at compile time.
     */
    template <Client... Clients>
    [[nodiscard]] constexpr __attribute__((always_inline)) AddrCounter for_clients() const
    {
        static_assert(sizeof...(Clients) != 0, "select at least one address-counter client");
        static_assert(((Clients == Client::Unpacker0 || Clients == Client::Unpacker1 || Clients == Client::Packers) && ...), "invalid address-counter client");
        auto result    = *this;
        result.clients = (std::uint8_t {0} | ... | static_cast<std::uint8_t>(Clients));
        return result;
    }

    /**
     * @brief Encode exactly one instruction without issuing it.
     *
     * During constant evaluation, selection follows set<Descriptor>(). Otherwise, selection uses
     * only the action and supplied fields, as in the runtime set() overload.
     *
     * @return One encoded 32-bit Tensix instruction, suitable for an instruction emitter or MOP slot.
     * @note Supply valid clients and action, select at least one counter, and supply values that
     *       fit the selected instruction. Use set<Descriptor>() for updates needing multiple instructions.
     * @note Invalid values and multi-instruction updates are rejected during constant evaluation.
     *       Runtime checks use LLK_ASSERT; the preconditions still apply when assertions are disabled.
     */
    [[nodiscard]] constexpr std::uint32_t operation() const;
};

namespace detail
{

inline constexpr std::uint8_t EmptyClientMask = 0; ///< No destination clients selected.
/** @brief Bitmask of all supported unpacker and packer clients. */
inline constexpr std::uint8_t AllClientsMask =
    static_cast<std::uint8_t>(Client::Unpacker0) | static_cast<std::uint8_t>(Client::Unpacker1) | static_cast<std::uint8_t>(Client::Packers);
inline constexpr std::uint8_t EmptyCounterMask = 0; ///< No counters selected within a group.
/** @brief X/Y group mask selecting channel 0 X at bit 0 and channel 1 X at bit 2. */
inline constexpr std::uint8_t BothXChannelsMask = 0b0101;

inline constexpr std::uint32_t PackedCounterValueMax    = 0x7u;    ///< Maximum three-bit packed assignment or increment.
inline constexpr std::uint32_t DualXCounterValueMax     = 0x3ffu;  ///< Maximum ten-bit X value in SETADCXX.
inline constexpr std::uint32_t SingleXCounterValueMax   = 0xffffu; ///< Maximum SETADC X value without a thread override.
inline constexpr std::uint32_t SingleYZWCounterValueMax = 0x1fffu; ///< Maximum value stored by a thirteen-bit Y, Z, or W counter.

/** @brief Four counter slots in either the X/Y or Z/W instruction group. */
class Group
{
public:
    Value fields[4]; ///< Channel 0 first/second dimension, followed by channel 1 first/second dimension.

    /**
     * @brief Encode field presence independently of the supplied values.
     *
     * @return A four-bit mask whose bit i selects fields[i]; explicitly supplied zeroes remain selected.
     */
    inline __attribute__((always_inline)) constexpr std::uint8_t mask() const
    {
        return fields[0].selected | (fields[1].selected << 1) | (fields[2].selected << 2) | (fields[3].selected << 3);
    }

    /**
     * @brief Read a slot's value, encoding omitted slots as zero.
     *
     * @param index: Slot index in [0, 3].
     * @return The supplied value for a selected slot, or zero for an omitted slot.
     */
    inline __attribute__((always_inline)) constexpr std::uint32_t value(const std::size_t index) const
    {
        return fields[index].selected ? fields[index].value : 0;
    }
};

/** @brief Instruction sequence with capacity for one SETADC per counter across both channels. */
class Sequence
{
public:
    std::uint32_t words[8] {}; ///< Encoded instructions in issue order; only the first count entries are used.
    std::size_t count = 0;     ///< Number of encoded instructions, in [0, 8].

    /**
     * @brief Append one encoded instruction.
     *
     * @param word: Instruction word to issue after the current sequence.
     * @note Ensure words has space for one more instruction.
     */
    constexpr void append(const std::uint32_t word)
    {
        words[count++] = word;
    }

    /**
     * @brief Append another sequence without changing its instruction order.
     *
     * @param other: Sequence to append; must not refer to this object.
     * @note Keep the combined count within the capacity of words.
     */
    constexpr void append(const Sequence& other)
    {
        for (std::size_t i = 0; i < other.count; ++i)
        {
            append(other.words[i]);
        }
    }
};

/**
 * @brief Test whether a group selects exactly one counter.
 *
 * @param mask: Counter-selection mask.
 * @return True if exactly one bit is set; false for an empty or multi-counter mask.
 */
inline __attribute__((always_inline)) constexpr bool single_bit(const std::uint8_t mask)
{
    return mask != EmptyCounterMask && (mask & (mask - 1)) == EmptyCounterMask;
}

/**
 * @brief Check descriptor structure before selecting instructions.
 *
 * @param descriptor: Update to validate.
 * @return True if the clients and action are valid and at least one counter is selected.
 * @note Value ranges are checked separately by the selected instruction encoders.
 */
inline __attribute__((always_inline)) constexpr bool is_valid(const AddrCounter& descriptor)
{
    return descriptor.clients != EmptyClientMask && descriptor.clients <= AllClientsMask &&
           (descriptor.action == Action::Set || descriptor.action == Action::Increment || descriptor.action == Action::AdvanceAndReset) &&
           (descriptor.channel0.x.selected || descriptor.channel0.y.selected || descriptor.channel0.z.selected || descriptor.channel0.w.selected ||
            descriptor.channel1.x.selected || descriptor.channel1.y.selected || descriptor.channel1.z.selected || descriptor.channel1.w.selected);
}

/**
 * @brief Reject an invalid constant expression or diagnose an invalid runtime request.
 *
 * @param valid: Condition required by the descriptor or instruction encoding.
 * @note False makes constant evaluation fail. Runtime checking uses LLK_ASSERT and is omitted
 *       when LLK assertions are disabled; callers must still satisfy the condition.
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
            // Keep assertion assembly outside the C++17 constexpr body. Valid constants never
            // enter this path, so operation() can fold even when runtime assertions are enabled.
            [valid] { LLK_ASSERT(valid, "invalid address-counter descriptor or value outside the selected instruction's field width"); }();
        }
    }
}

/**
 * @brief Encode a full-width assignment to one counter with SETADC.
 *
 * @param clients: Destination client bitmask.
 * @param zw: True for the Z/W group; false for X/Y.
 * @param index: Group slot in [0, 3], identifying the channel and dimension.
 * @param value: Assignment in [0, 65535] for X or [0, 8191] for Y/Z/W; thread-override bits are excluded.
 * @return The encoded SETADC instruction.
 */
inline __attribute__((always_inline)) constexpr std::uint32_t single(
    const std::uint8_t clients, const bool zw, const std::size_t index, const std::uint32_t value)
{
    const auto channel   = static_cast<std::uint32_t>(index / 2);
    const auto dimension = static_cast<std::uint32_t>((zw ? 2 : 0) + index % 2);
    // SETADC's top two operand bits select a thread; Y/Z/W storage is narrower still.
    const auto value_max = dimension == 0 ? SingleXCounterValueMax : SingleYZWCounterValueMax;
    require(value <= value_max && TT_SETADC_VALID(clients, channel, dimension, value));
    return TT_OP_SETADC(clients, channel, dimension, value);
}

/**
 * @brief Encode both X assignments with SETADCXX, placing X1 before X0 in the operand encoding.
 *
 * @param clients: Destination client bitmask.
 * @param group: X/Y group supplying channel 0 X and channel 1 X; Y fields are ignored.
 * @return The encoded SETADCXX instruction.
 * @note Select both X counters and supply each value in [0, 1023].
 *       SETADCXX also writes both X CR shadows, as modeled by the
 *       [Blackhole simulator](https://github.com/tenstorrent/ttsim/blob/464814f86583af9def3cf1643e58236ead759ae9/src/tensix.cpp#L3501-L3512).
 */
inline __attribute__((always_inline)) constexpr std::uint32_t xx(const std::uint8_t clients, const Group& group)
{
    // SETADCXX takes Channel1 first, then Channel0.
    require(group.value(2) <= DualXCounterValueMax && TT_SETADCXX_VALID(clients, group.value(2), group.value(0)));
    return TT_OP_SETADCXX(clients, group.value(2), group.value(0));
}

/**
 * @brief Encode one three-bit packed set, live increment, or CR advance-and-reload instruction.
 *
 * @param clients: Destination client bitmask.
 * @param zw: True for the Z/W group; false for X/Y.
 * @param action: Select SETADCXY/ZW, INCADCXY/ZW, or ADDRCRXY/ZW.
 * @param group: Counter selection and values; each selected value must be in [0, 7].
 * @return The encoded instruction for the requested group and action.
 * @note Set and AdvanceAndReset use a write mask. Increment has no mask: omitted slots encode
 *       zero increments, preserving their live counters and CR shadows.
 */
inline __attribute__((always_inline)) constexpr std::uint32_t packed(const std::uint8_t clients, const bool zw, const Action action, const Group& group)
{
    const auto a    = group.value(0);
    const auto b    = group.value(1);
    const auto c    = group.value(2);
    const auto d    = group.value(3);
    const auto mask = group.mask();
    // All counter values are three bits. Higher bits of the final slot encode a thread override.
    require(d <= PackedCounterValueMax && TT_SETADCXY_VALID(clients, d, c, b, a, mask));
    if (action == Action::Set)
    {
        return zw ? TT_OP_SETADCZW(clients, d, c, b, a, mask) : TT_OP_SETADCXY(clients, d, c, b, a, mask);
    }
    if (action == Action::Increment)
    {
        return zw ? TT_OP_INCADCZW(clients, d, c, b, a) : TT_OP_INCADCXY(clients, d, c, b, a);
    }
    return zw ? TT_OP_ADDRCRZW(clients, d, c, b, a, mask) : TT_OP_ADDRCRXY(clients, d, c, b, a, mask);
}

/**
 * @brief Encode one SETADC for a singleton, otherwise one masked packed set.
 *
 * @param clients: Destination client bitmask.
 * @param zw: True for the Z/W group; false for X/Y.
 * @param group: Nonempty counter selection.
 * @return One encoded SETADC or SETADCXY/ZW instruction.
 * @note Supply singleton X values in [0, 65535] or Y/Z/W values in [0, 8191].
 *       Supply every value in a multi-counter group in [0, 7].
 */
inline __attribute__((always_inline)) constexpr std::uint32_t set_group(const std::uint8_t clients, const bool zw, const Group& group)
{
    if (single_bit(group.mask()))
    {
        for (std::size_t i = 0; i < 4; ++i)
        {
            if (group.fields[i].selected)
            {
                return single(clients, zw, i, group.value(i));
            }
        }
    }
    return packed(clients, zw, Action::Set, group);
}

/**
 * @brief Select one runtime instruction using only the action and supplied fields.
 *
 * @param clients: Destination client bitmask.
 * @param zw: True for the Z/W group; false for X/Y.
 * @param action: Counter effect to encode.
 * @param group: Nonempty counter selection with values fitting the selected instruction.
 * @return One encoded instruction; a Set of X0 and X1 alone uses SETADCXX.
 * @note Runtime values never trigger a fallback to a different instruction or a split sequence.
 */
inline constexpr __attribute__((always_inline)) std::uint32_t runtime_operation(
    const std::uint8_t clients, const bool zw, const Action action, const Group& group)
{
    if (action != Action::Set)
    {
        return packed(clients, zw, action, group);
    }
    return !zw && group.mask() == BothXChannelsMask ? xx(clients, group) : set_group(clients, zw, group);
}

/**
 * @brief Pack every small constant assignment together; use SETADC for larger values.
 *
 * @param clients: Destination client bitmask.
 * @param zw: True for the Z/W group; false for X/Y.
 * @param group: Counter assignments in [0, 65535] for X or [0, 8191] for Y/Z/W.
 * @return Wide singleton writes followed by at most one instruction for values in [0, 7].
 * @note Empty groups produce an empty sequence. SETADCXX is considered separately by plan_group().
 */
constexpr Sequence constant_sets(const std::uint8_t clients, const bool zw, const Group& group)
{
    Sequence result;
    Group small {};
    for (std::size_t i = 0; i < 4; ++i)
    {
        if (group.fields[i].selected)
        {
            if (group.value(i) <= PackedCounterValueMax)
            {
                small.fields[i] = group.fields[i];
            }
            else
            {
                result.append(single(clients, zw, i, group.value(i)));
            }
        }
    }
    if (small.mask() != EmptyCounterMask)
    {
        result.append(set_group(clients, zw, small));
    }
    return result;
}

/**
 * @brief Plan the instructions for one X/Y or Z/W group.
 *
 * @param clients: Destination client bitmask.
 * @param zw: True for the Z/W group; false for X/Y.
 * @param action: Counter effect to encode.
 * @param group: Sparse counter values for the selected group.
 * @return An empty sequence for an omitted group, otherwise the instructions implementing the update.
 * @note Constant Set updates use the fewest instructions, preferring SETADCXX on equal counts.
 *       Runtime updates, Increment, and AdvanceAndReset use one instruction per nonempty group.
 */
constexpr Sequence plan_group(const std::uint8_t clients, const bool zw, const Action action, const Group& group)
{
    Sequence result;
    const auto mask = group.mask();
    if (mask == EmptyCounterMask)
    {
        return result;
    }
    if (action != Action::Set)
    {
        result.append(packed(clients, zw, action, group));
    }
    else if (__builtin_is_constant_evaluated())
    {
        result = constant_sets(clients, zw, group);
        if (!zw && (mask & BothXChannelsMask) == BothXChannelsMask && group.value(2) <= DualXCounterValueMax &&
            TT_SETADCXX_VALID(clients, group.value(2), group.value(0)))
        {
            Sequence with_xx;
            with_xx.append(xx(clients, group));
            const Group ys {{Value {}, group.fields[1], Value {}, group.fields[3]}};
            with_xx.append(constant_sets(clients, false, ys));
            // Prefer the dedicated dual-X instruction whenever it needs no additional instructions.
            if (with_xx.count <= result.count)
            {
                result = with_xx;
            }
        }
    }
    else
    {
        // Runtime values have the same field-width preconditions as the raw instruction.
        // Selection depends only on the supplied fields, never on their runtime values.
        result.append(runtime_operation(clients, zw, action, group));
    }
    return result;
}

/**
 * @brief Validate and plan an entire counter update without issuing instructions.
 *
 * @param descriptor: Sparse counter update and destination clients.
 * @return The X/Y sequence followed by the Z/W sequence, containing at most eight instructions.
 * @note No supported instruction spans both groups, so their instruction counts add.
 */
constexpr Sequence plan(const AddrCounter& descriptor)
{
    require(is_valid(descriptor));
    const Group xy {{descriptor.channel0.x, descriptor.channel0.y, descriptor.channel1.x, descriptor.channel1.y}};
    const Group zw {{descriptor.channel0.z, descriptor.channel0.w, descriptor.channel1.z, descriptor.channel1.w}};
    auto result = plan_group(descriptor.clients, false, descriptor.action, xy);
    result.append(plan_group(descriptor.clients, true, descriptor.action, zw));
    return result;
}

/**
 * @brief Issue one compile-time instruction through TTI_INSN.
 *
 * @tparam Word: Validated 32-bit instruction encoding.
 */
template <std::uint32_t Word>
inline __attribute__((always_inline)) void emit()
{
    TTI_INSN(Word);
}

/**
 * @brief Issue selected words of a compile-time descriptor in index order.
 *
 * @tparam Descriptor: Valid counter update whose plan is evaluated at compile time.
 * @tparam Indices: Valid zero-based instruction indices, normally the entire planned sequence.
 */
template <auto Descriptor, std::size_t... Indices>
inline __attribute__((always_inline)) void emit(const std::index_sequence<Indices...>)
{
    constexpr auto sequence = plan(Descriptor);
    (emit<sequence.words[Indices]>(), ...);
}

/**
 * @brief Issue one runtime-encoded instruction for a nonempty counter group.
 *
 * @tparam ZW: True for the Z/W group; false for X/Y.
 * @param clients: Destination client bitmask.
 * @param action: Counter effect to issue.
 * @param group: Sparse values satisfying runtime_operation()'s encoding limits.
 * @note An empty group issues nothing; a nonempty group is written through TT_INSN.
 */
template <bool ZW>
inline __attribute__((always_inline)) void emit_runtime(const std::uint8_t clients, const Action action, const Group& group)
{
    if (group.mask() != EmptyCounterMask)
    {
        TT_INSN(runtime_operation(clients, ZW, action, group));
    }
}

} // namespace detail

inline constexpr __attribute__((always_inline)) std::uint32_t AddrCounter::operation() const
{
    const auto sequence = detail::plan(*this);
    detail::require(sequence.count == 1);
    return sequence.words[0];
}

/**
 * @brief Issue a compile-time counter update using immediate instructions.
 *
 * @tparam Descriptor: Sparse assignments and client mask; action selects set, increment, or carriage return.
 * @note Select at least one valid client and one counter, and supply a valid action.
 *       Supply Set values in [0, 65535] for X or [0, 8191] for Y/Z/W.
 *       Supply Increment and AdvanceAndReset values in [0, 7].
 * @note Invalid descriptors are rejected at compile time. Set uses the fewest supported instructions,
 *       combining three-bit packed writes, ten-bit SETADCXX writes, and dimension-limited SETADC writes.
 *       Equal instruction counts favor SETADCXX for two selected X counters. X/Y instructions
 *       precede Z/W instructions. REG2FLOP and thread overrides are not used.
 */
template <auto Descriptor>
inline __attribute__((always_inline)) void set()
{
    static_assert(detail::is_valid(Descriptor), "address-counter descriptor requires valid clients, an action, and at least one field");
    constexpr auto sequence = detail::plan(Descriptor);
    detail::emit<Descriptor>(std::make_index_sequence<sequence.count> {});
}

/**
 * @brief Issue a runtime-valued descriptor through the instruction buffer.
 *
 * @param descriptor: Sparse assignments and client mask, with values fitting the selected instructions.
 * @note Prefer set<Descriptor>() when values are constant: it validates and splits updates at compile time.
 *       For example, channel0 = {.x = 10, .y = 2} needs two instructions and is rejected by this overload's
 *       LLK_ASSERT checks, whereas set<Descriptor>() issues both instructions.
 * @note Select at least one valid client and one counter, and supply a valid action. Within each
 *       X/Y or Z/W group, supply singleton Set values in [0, 65535] for X or [0, 8191] for Y/Z/W,
 *       a Set of X0+X1 alone in [0, 1023] per X, and other Set selections in [0, 7] per value.
 *       Supply Increment and AdvanceAndReset values in [0, 7].
 * @note Issues one instruction per nonempty group, X/Y before Z/W. Supply constant action, clients,
 *       and field presence to let selection fold away. Values do not trigger instruction splitting:
 *       wider assignments requiring SETADC must be supplied as separate singleton descriptors.
 *       Validation uses LLK_ASSERT; the preconditions still apply when assertions are disabled.
 *       Thread overrides are not supported.
 */
inline __attribute__((always_inline)) void set(const AddrCounter descriptor)
{
    detail::require(detail::is_valid(descriptor));
    const detail::Group xy {{descriptor.channel0.x, descriptor.channel0.y, descriptor.channel1.x, descriptor.channel1.y}};
    const detail::Group zw {{descriptor.channel0.z, descriptor.channel0.w, descriptor.channel1.z, descriptor.channel1.w}};
    detail::emit_runtime<false>(descriptor.clients, descriptor.action, xy);
    detail::emit_runtime<true>(descriptor.clients, descriptor.action, zw);
}

} // namespace hal::counters::adc
