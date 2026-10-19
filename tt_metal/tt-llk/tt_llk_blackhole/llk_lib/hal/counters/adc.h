// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "../utils/gpr.h"
#include "ckernel.h"
#include "llk_assert.h"

namespace hal::counters::adc
{

/** @brief Select address counters belonging to either unpacker or all four packers. */
enum class Client : std::uint8_t
{
    Unpacker0 = 0x1,
    Unpacker1 = 0x2,
    Packers   = 0x4
};

/** @brief Select the effect on each explicitly supplied counter. */
enum class Action : std::uint8_t
{
    Set,            // Replace both the live counter and its carry/reset shadow.
    Increment,      // Increment the live counter; preserve its shadow.
    AdvanceAndReset // Increment the shadow, then copy it to the live counter.
};

/**
 * @brief An optional counter value. Omission preserves the counter; an explicit zero selects it.
 *
 * Public members keep descriptors usable as template arguments. Normally initialize a field
 * directly with its value, for example .x = 10. GPR sources are rejected on Blackhole.
 */
struct Value
{
    std::uint32_t value = 0;
    bool selected       = false;

    constexpr Value() = default;

    constexpr __attribute__((always_inline)) Value(const std::uint32_t value) : value(value), selected(true)
    {
    }

    template <std::uint32_t Index>
    constexpr Value(const hal::Gpr<Index>)
    {
        static_assert(Index != Index, "Blackhole does not handle REG2FLOP properly. GPR address-counter writes are unsupported");
    }
};

/** @brief Sparse X/Y/Z/W values for one address-counter channel. */
struct Channel
{
    Value x {};
    Value y {};
    Value z {};
    Value w {};
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
 * @code
 * constexpr AddrCounter matmul = AddrCounter {
 *     .channel0 = {.x = 10, .y = 20},
 *     .channel1 = {.y = 0},
 * }.for_clients<Client::Unpacker0, Client::Unpacker1>();
 * set<matmul>();
 * @endcode
 */
struct AddrCounter
{
    Action action = Action::Set;
    Channel channel0 {};
    Channel channel1 {};
    std::uint8_t clients = 0;

    /** @brief Return the same descriptor with its destination client mask replaced. */
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
     * Constant descriptors use the same instruction selection as set<Descriptor>(). Descriptions
     * needing multiple instructions are rejected. At runtime, values must fit the instruction
     * selected by the supplied fields; enable LLK assertions to diagnose invalid values.
     */
    [[nodiscard]] constexpr std::uint32_t operation() const;
};

namespace detail
{

inline constexpr std::uint8_t EmptyClientMask = 0;
inline constexpr std::uint8_t AllClientsMask =
    static_cast<std::uint8_t>(Client::Unpacker0) | static_cast<std::uint8_t>(Client::Unpacker1) | static_cast<std::uint8_t>(Client::Packers);
inline constexpr std::uint8_t EmptyCounterMask = 0;
// Within an X/Y group, bit 0 selects Channel0 X and bit 2 selects Channel1 X.
inline constexpr std::uint8_t BothXChannelsMask = 0b0101;

inline constexpr std::uint32_t PackedCounterValueMax = 0x7u;
inline constexpr std::uint32_t DualXCounterValueMax  = 0x3ffu;
inline constexpr std::uint32_t SingleCounterValueMax = 0xffffu;

struct Group
{
    Value fields[4]; // Channel0 first dimension, Channel0 second, Channel1 first, Channel1 second.

    inline __attribute__((always_inline)) constexpr std::uint8_t mask() const
    {
        return fields[0].selected | (fields[1].selected << 1) | (fields[2].selected << 2) | (fields[3].selected << 3);
    }

    inline __attribute__((always_inline)) constexpr std::uint32_t value(const std::size_t index) const
    {
        return fields[index].selected ? fields[index].value : 0;
    }
};

struct Sequence
{
    std::uint32_t words[8] {};
    std::size_t count = 0;

    constexpr void append(const std::uint32_t word)
    {
        words[count++] = word;
    }

    constexpr void append(const Sequence& other)
    {
        for (std::size_t i = 0; i < other.count; ++i)
        {
            append(other.words[i]);
        }
    }
};

inline __attribute__((always_inline)) constexpr bool single_bit(const std::uint8_t mask)
{
    return mask != EmptyCounterMask && (mask & (mask - 1)) == EmptyCounterMask;
}

inline __attribute__((always_inline)) constexpr bool is_valid(const AddrCounter& descriptor)
{
    return descriptor.clients != EmptyClientMask && descriptor.clients <= AllClientsMask &&
           (descriptor.action == Action::Set || descriptor.action == Action::Increment || descriptor.action == Action::AdvanceAndReset) &&
           (descriptor.channel0.x.selected || descriptor.channel0.y.selected || descriptor.channel0.z.selected || descriptor.channel0.w.selected ||
            descriptor.channel1.x.selected || descriptor.channel1.y.selected || descriptor.channel1.z.selected || descriptor.channel1.w.selected);
}

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

inline __attribute__((always_inline)) constexpr std::uint32_t single(
    const std::uint8_t clients, const bool zw, const std::size_t index, const std::uint32_t value)
{
    const auto channel   = static_cast<std::uint32_t>(index / 2);
    const auto dimension = static_cast<std::uint32_t>((zw ? 2 : 0) + index % 2);
    // The top two bits accepted by the raw macro select a thread, not a counter value.
    require(value <= SingleCounterValueMax && TT_SETADC_VALID(clients, channel, dimension, value));
    return TT_OP_SETADC(clients, channel, dimension, value);
}

inline __attribute__((always_inline)) constexpr std::uint32_t xx(const std::uint8_t clients, const Group& group)
{
    // SETADCXX takes Channel1 first, then Channel0.
    require(group.value(2) <= DualXCounterValueMax && TT_SETADCXX_VALID(clients, group.value(2), group.value(0)));
    return TT_OP_SETADCXX(clients, group.value(2), group.value(0));
}

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

/** @brief Encode one SETADC for a singleton, otherwise one masked packed set. */
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

/** @brief Select one runtime instruction using only the action and supplied fields. */
inline constexpr __attribute__((always_inline)) std::uint32_t runtime_operation(
    const std::uint8_t clients, const bool zw, const Action action, const Group& group)
{
    if (action != Action::Set)
    {
        return packed(clients, zw, action, group);
    }
    return !zw && group.mask() == BothXChannelsMask ? xx(clients, group) : set_group(clients, zw, group);
}

/** @brief Pack every small constant assignment together; use SETADC for larger values. */
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

constexpr Sequence plan(const AddrCounter& descriptor)
{
    require(is_valid(descriptor));
    const Group xy {{descriptor.channel0.x, descriptor.channel0.y, descriptor.channel1.x, descriptor.channel1.y}};
    const Group zw {{descriptor.channel0.z, descriptor.channel0.w, descriptor.channel1.z, descriptor.channel1.w}};
    auto result = plan_group(descriptor.clients, false, descriptor.action, xy);
    result.append(plan_group(descriptor.clients, true, descriptor.action, zw));
    return result;
}

template <std::uint32_t Word>
inline __attribute__((always_inline)) void emit()
{
    TTI_INSN(Word);
}

template <auto Descriptor, std::size_t... Indices>
inline __attribute__((always_inline)) void emit(const std::index_sequence<Indices...>)
{
    constexpr auto sequence = plan(Descriptor);
    (emit<sequence.words[Indices]>(), ...);
}

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
 * @brief Issue a constant descriptor using the fewest supported instructions.
 *
 * @tparam Descriptor: Sparse assignments and client mask; action selects set, increment, or carriage return.
 * @note Set may combine three-bit packed writes, ten-bit SETADCXX writes, and sixteen-bit SETADC
 *       writes. Increment and carriage-return values must fit three bits. REG2FLOP is never used.
 *       When minimum-length set sequences tie, prefer SETADCXX for the two selected X counters.
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
 * @note Supply constant action, clients, and field presence for selection to fold away. A singleton
 *       uses sixteen-bit SETADC, X0+X1 alone uses ten-bit SETADCXX, and other sets use three-bit
 *       SETADCXY/ZW. Larger values needing SETADC must be supplied as separate singleton descriptors.
 *       Enable LLK assertions for range checks. Thread overrides are not supported.
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
