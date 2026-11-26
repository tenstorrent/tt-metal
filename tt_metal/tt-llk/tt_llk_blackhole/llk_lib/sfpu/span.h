// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel_addrmod.h"
#include "sfpi.h"

// NOTE: Experimental, use with care

namespace llk::sfpu
{

template <typename T, std::size_t Size>
class vArray;

template <std::size_t Size>
class vSpan;

namespace detail
{

template <typename F, std::size_t... I>
[[gnu::always_inline]] inline void unroll(F&& f, std::index_sequence<I...>)
{
    (f(I), ...);
}

/// @brief Calls f(0), f(1), ..., f(Size - 1), fully unrolled.
template <std::size_t Size, typename F>
[[gnu::always_inline]] inline void unroll(F&& f)
{
    unroll(f, std::make_index_sequence<Size> {});
}

/// A number or an SFPI vector used as an operand of a lane by lane operation: the same value in every lane.
template <typename T>
class Scalar
{
    T scalar;

public:
    static constexpr bool is_vector()
    {
        return false;
    }

    [[gnu::always_inline]] explicit Scalar(const T& value) : scalar(value)
    {
    }

    /// The value, whichever lane is asked for
    [[gnu::always_inline]] const T& value(std::size_t) const
    {
        return scalar;
    }
};

/// Base of vArray and vSpan: Size lanes, lane i read with value(i).
template <typename Derived, std::size_t Size>
class Vector
{
    static_assert(Size > 0, "a vector needs at least one lane");

protected:
    Vector() = default;

public:
    static constexpr bool is_vector()
    {
        return true;
    }

    static constexpr std::size_t lanes()
    {
        return Size;
    }

    [[gnu::always_inline]] decltype(auto) value(const std::size_t i) const
    {
        return static_cast<const Derived&>(*this)[i];
    }
};

template <typename Derived, std::size_t Size>
[[gnu::always_inline]] inline const Vector<Derived, Size>& operand(const Vector<Derived, Size>& vector)
{
    return vector;
}

/// @brief Matches if SFPI can make a vector out of T
template <typename T>
constexpr bool is_broadcastable_v = std::is_constructible_v<sfpi::vFloat, const T&> || std::is_constructible_v<sfpi::vInt, const T&> ||
                                    std::is_constructible_v<sfpi::vUInt, const T&> || std::is_constructible_v<sfpi::vSMag, const T&>;

template <typename T, std::enable_if_t<is_broadcastable_v<T>, int> = 0>
[[gnu::always_inline]] inline Scalar<T> operand(const T& value)
{
    return Scalar<T>(value);
}

/// @brief Lanes of an operand, 0 for a Scalar, which fits any number of them.
template <typename Operand>
constexpr std::size_t lanes_of()
{
    if constexpr (Operand::is_vector())
    {
        return Operand::lanes();
    }
    else
    {
        return 0;
    }
}

/// @brief Whether the operands can be walked together: at least one is a Vector, and all Vectors have the same lanes.
template <typename... Operand>
constexpr bool compatible()
{
    constexpr std::size_t lanes = std::max({lanes_of<Operand>()...});
    return lanes != 0 && ((lanes_of<Operand>() == 0 || lanes_of<Operand>() == lanes) && ...);
}

template <typename Functor, typename... Operand>
[[gnu::always_inline]] inline auto map(Functor& functor, const Operand&... operand)
{
    static_assert(compatible<Operand...>(), "operands must be compatible: at least one vArray or vSpan, all with the same number of lanes");

    constexpr std::size_t Size = std::max({lanes_of<Operand>()...});
    using Result               = std::decay_t<decltype(functor(operand.value(0)...))>;

    static_assert(
        std::is_void_v<Result> || (!std::is_arithmetic_v<Result> && !std::is_same_v<Result, std::decay_t<decltype(sfpi::dst_reg[0])>>),
        "the functor must return an SFPI vector (vFloat, vInt, vUInt, ...) or nothing");

    if constexpr (std::is_void_v<Result>)
    {
        unroll<Size>([&](const std::size_t i) { functor(operand.value(i)...); });
    }
    else
    {
        return vArray<Result, Size>::generate([&](const std::size_t i) { return functor(operand.value(i)...); });
    }
}

} // namespace detail

/**
 * @brief Apply @p functor over all lanes of the operands
 *
 * The escape hatch for anything without a vArray overload (sfpi::abs, sfpi::setexp, v_if, ...)
 *
 * @param functor: Operation over lanes
 * @param operands: vArrays and vSpans or Scalars
 * @return vArray of what @p functor returns, nothing if it returns void
 */
template <typename Functor, typename... Operand>
[[gnu::always_inline]] inline auto map(Functor&& functor, const Operand&... operands)
{
    return detail::map(functor, detail::operand(operands)...);
}

/**
 * @brief View over Size consecutive Vectors in Dest
 *
 * Like sfpi::dst_reg[i], for Size vectors:
 * - converting to a vArray loads
 * - assigning to a vSpan stores.
 *
 * @tparam Size: Number of lanes
 */
template <std::size_t Size>
class vSpan : public detail::Vector<vSpan<Size>, Size>
{
    int offset; // In Dest vectors from the RWC
    int addr_mode;

    [[gnu::always_inline]] constexpr vSpan(const int offset, const int addr_mode) : offset(offset), addr_mode(addr_mode)
    {
    }

public:
    /// @brief Size consecutive Dest vectors, the first @p offset vectors from the Dest RWC.
    [[gnu::always_inline]] static constexpr vSpan fixed(const int offset)
    {
        return vSpan(offset, ckernel::ADDR_MOD_7);
    }

    /**
     * @brief Output view at @p offset whose every store advances Dest by one vector
     *
     * @warning Should only be used for final store, because it advances Dest RWC
     */
    [[gnu::always_inline]] static constexpr vSpan advancing(const int offset)
    {
        return vSpan(offset, ckernel::ADDR_MOD_6);
    }

    vSpan(const vSpan&) = default;

    // Would rebind the view rather than copy Dest, which is confusing
    vSpan& operator=(const vSpan&) = delete;

    [[gnu::always_inline]] auto operator[](const std::size_t i) const
    {
        const int lane = addr_mode == ckernel::ADDR_MOD_7 ? static_cast<int>(i) : 0;
        return sfpi::dst_reg[offset + lane].mode(addr_mode);
    }

    /// @note Store to vSpan writes to Dest
    template <typename Value>
    [[gnu::always_inline]] void operator=(const Value& value) const
    {
        const auto& source = detail::operand(value);

        static_assert(
            detail::compatible<vSpan, std::decay_t<decltype(source)>>(),
            "operands must be compatible: at least one vArray or vSpan, all with the same number of lanes");

        detail::unroll<Size>([&](const std::size_t i) { (*this)[i] = source.value(i); });
    }
};

/**
 * @brief Size independent SFPI vectors that are logically grouped in operations.
 *
 * @tparam T: SFPI vector type of each lane (vFloat, vInt, vUInt, ...)
 * @tparam Size: Number of lanes
 */
template <typename T, std::size_t Size>
class vArray : public detail::Vector<vArray<T, Size>, Size>
{
    T values[Size];

    // Lanes are constructed in place, in order: vBool and comparison results have no default constructor.
    template <std::size_t... I, typename Generator>
    [[gnu::always_inline]] vArray(std::index_sequence<I...>, Generator&& generator) : values {T(generator(I))...}
    {
    }

    template <typename Right, typename Assign>
    [[gnu::always_inline]] vArray& lanewise(const Right& right, Assign&& assign)
    {
        const auto& source = detail::operand(right);
        detail::unroll<Size>([&](const std::size_t i) { assign(values[i], source.value(i)); });
        return *this;
    }

public:
    using vType = T;

    vArray() = default;

    /// Every lane = T(value), each in its own LReg; operators take a scalar directly and share one LReg across lanes.
    template <typename U, std::enable_if_t<std::is_constructible_v<T, const U&>, int> = 0>
    [[gnu::always_inline]] explicit vArray(const U& value) :
        vArray(std::make_index_sequence<Size> {}, [&](std::size_t) -> const U& { return value; })
    {
    }

    /// Lane i = T(span[i]): loads every lane
    [[gnu::always_inline]] vArray(const vSpan<Size>& span) : vArray(std::make_index_sequence<Size> {}, [&](std::size_t i) { return span[i]; })
    {
    }

    /// Lane i = T(other[i])
    template <typename U, std::enable_if_t<!std::is_same_v<U, T> && std::is_constructible_v<T, const U&>, int> = 0>
    [[gnu::always_inline]] explicit vArray(const vArray<U, Size>& other) :
        vArray(std::make_index_sequence<Size> {}, [&](std::size_t i) -> const U& { return other[i]; })
    {
    }

    /// Lane i = T(generator(i))
    template <typename Generator>
    [[gnu::always_inline]] static vArray generate(Generator&& generator)
    {
        return vArray(std::make_index_sequence<Size> {}, generator);
    }

    [[gnu::always_inline]] T& operator[](const std::size_t i)
    {
        return values[i];
    }

    [[gnu::always_inline]] const T& operator[](const std::size_t i) const
    {
        return values[i];
    }

    template <typename U = T, typename = decltype(std::declval<U&>() += std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator+=(const vArray& right)
    {
        return lanewise(right, [](T& l, const T& r) { l += r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() += std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator+=(const T& right)
    {
        return lanewise(right, [](T& l, const T& r) { l += r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() -= std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator-=(const vArray& right)
    {
        return lanewise(right, [](T& l, const T& r) { l -= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() -= std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator-=(const T& right)
    {
        return lanewise(right, [](T& l, const T& r) { l -= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() *= std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator*=(const vArray& right)
    {
        return lanewise(right, [](T& l, const T& r) { l *= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() *= std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator*=(const T& right)
    {
        return lanewise(right, [](T& l, const T& r) { l *= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() &= std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator&=(const vArray& right)
    {
        return lanewise(right, [](T& l, const T& r) { l &= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() &= std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator&=(const T& right)
    {
        return lanewise(right, [](T& l, const T& r) { l &= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() |= std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator|=(const vArray& right)
    {
        return lanewise(right, [](T& l, const T& r) { l |= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() |= std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator|=(const T& right)
    {
        return lanewise(right, [](T& l, const T& r) { l |= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() ^= std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator^=(const vArray& right)
    {
        return lanewise(right, [](T& l, const T& r) { l ^= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() ^= std::declval<const U&>())>
    [[gnu::always_inline]] vArray& operator^=(const T& right)
    {
        return lanewise(right, [](T& l, const T& r) { l ^= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() <<= 1u)>
    [[gnu::always_inline]] vArray& operator<<=(const unsigned amount)
    {
        return lanewise(amount, [](T& l, const unsigned r) { l <<= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() <<= std::declval<const sfpi::vUInt&>())>
    [[gnu::always_inline]] vArray& operator<<=(const vArray<sfpi::vUInt, Size>& amount)
    {
        return lanewise(amount, [](T& l, const sfpi::vUInt& r) { l <<= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() >>= 1u)>
    [[gnu::always_inline]] vArray& operator>>=(const unsigned amount)
    {
        return lanewise(amount, [](T& l, const unsigned r) { l >>= r; });
    }

    template <typename U = T, typename = decltype(std::declval<U&>() >>= std::declval<const sfpi::vUInt&>())>
    [[gnu::always_inline]] vArray& operator>>=(const vArray<sfpi::vUInt, Size>& amount)
    {
        return lanewise(amount, [](T& l, const sfpi::vUInt& r) { l >>= r; });
    }

    template <typename U = T, typename = decltype(-std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator-(const vArray& x)
    {
        return map([](const T& value) { return -value; }, x);
    }

    template <typename U = T, typename = decltype(~std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator~(const vArray& x)
    {
        return map([](const T& value) { return ~value; }, x);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() + std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator+(const vArray& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l + r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() + std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator+(const vArray& left, const T& right)
    {
        return map([](const T& l, const T& r) { return l + r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() + std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator+(const T& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l + r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() - std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator-(const vArray& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l - r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() - std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator-(const vArray& left, const T& right)
    {
        return map([](const T& l, const T& r) { return l - r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() - std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator-(const T& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l - r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() * std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator*(const vArray& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l * r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() * std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator*(const vArray& left, const T& right)
    {
        return map([](const T& l, const T& r) { return l * r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() * std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator*(const T& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l * r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() & std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator&(const vArray& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l & r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() & std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator&(const vArray& left, const T& right)
    {
        return map([](const T& l, const T& r) { return l & r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() & std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator&(const T& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l & r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() | std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator|(const vArray& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l | r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() | std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator|(const vArray& left, const T& right)
    {
        return map([](const T& l, const T& r) { return l | r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() | std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator|(const T& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l | r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() ^ std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator^(const vArray& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l ^ r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() ^ std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator^(const vArray& left, const T& right)
    {
        return map([](const T& l, const T& r) { return l ^ r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() ^ std::declval<const U&>())>
    [[gnu::always_inline]] friend auto operator^(const T& left, const vArray& right)
    {
        return map([](const T& l, const T& r) { return l ^ r; }, left, right);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() << 1u)>
    [[gnu::always_inline]] friend auto operator<<(const vArray& left, const unsigned amount)
    {
        return map([](const T& l, const unsigned r) { return l << r; }, left, amount);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() << std::declval<const sfpi::vUInt&>())>
    [[gnu::always_inline]] friend auto operator<<(const vArray& left, const vArray<sfpi::vUInt, Size>& amount)
    {
        return map([](const T& l, const sfpi::vUInt& r) { return l << r; }, left, amount);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() >> 1u)>
    [[gnu::always_inline]] friend auto operator>>(const vArray& left, const unsigned amount)
    {
        return map([](const T& l, const unsigned r) { return l >> r; }, left, amount);
    }

    template <typename U = T, typename = decltype(std::declval<const U&>() >> std::declval<const sfpi::vUInt&>())>
    [[gnu::always_inline]] friend auto operator>>(const vArray& left, const vArray<sfpi::vUInt, Size>& amount)
    {
        return map([](const T& l, const sfpi::vUInt& r) { return l >> r; }, left, amount);
    }
};

template <std::size_t Size>
using vFloatArray = vArray<sfpi::vFloat, Size>;

template <std::size_t Size>
using vIntArray = vArray<sfpi::vInt, Size>;

template <std::size_t Size>
using vUIntArray = vArray<sfpi::vUInt, Size>;

} // namespace llk::sfpu
