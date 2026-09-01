// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "access_types.h"
#include "detail/composition_traits.h"
#include "field.h"

namespace hal::cfg
{

/**
 * @brief Resolve the absolute config word of a field section at compile time.
 *
 * @tparam F: Field whose containing word is resolved.
 * @tparam S: Repeated descriptor section; compilation fails when it is outside F.count.
 */
template <const Field& F, Sec S>
inline constexpr std::uint32_t word_addr = []
{
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    return F.addr32(S);
}();

/**
 * @brief One field assignment, not yet written to hardware.
 *
 * Use @ref set to construct one. Assignments from different generated classes
 * can be combined when their fields occupy the same physical CFG word.
 */
template <const Field& F, Sec S>
class FieldAssignment
{
public:
    static_assert(F.width <= 32, "field wider than 32b cannot be assigned through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    static constexpr RegisterFile file   = F.file;
    static constexpr std::uint32_t addr  = F.addr32(S);
    static constexpr std::uint32_t shift = F.shamt(S);
    static constexpr std::uint32_t mask  = F.mask(S);

    std::uint32_t value;
};

/**
 * @brief One compile-time field assignment.
 *
 * Unlike @ref FieldAssignment, the value is part of the type. Combining only
 * constant assignments therefore emits immediate TTI_RMWCIB/TTI_SETC16
 * instructions without constructing an opcode at runtime.
 */
template <const Field& F, Sec S, std::uint32_t Value>
class ConstantFieldAssignment
{
public:
    static_assert(F.width <= 32, "field wider than 32b cannot be assigned through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(Value <= ((std::uint64_t {1} << F.width) - 1u), "value exceeds field width");

    static constexpr RegisterFile file   = F.file;
    static constexpr std::uint32_t addr  = F.addr32(S);
    static constexpr std::uint32_t shift = F.shamt(S);
    static constexpr std::uint32_t mask  = F.mask(S);
    static constexpr std::uint32_t value = Value;
};

/**
 * @brief One destination-bound whole-word GPR transfer.
 *
 * Use @ref from_gpr to construct one. Unlike a field assignment, this operation
 * replaces one or four complete state-CFG words and acts as an ordering barrier
 * between automatically grouped assignment runs.
 */
template <const Field& F, Sec S, typename Source>
class GprWrite
{
public:
    static_assert(F.file == RegisterFile::State, "GPR-backed CFG writes require a state-CFG destination");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) == 0, "GPR-backed CFG writes must start at the beginning of a CFG word");

    static constexpr RegisterFile file   = F.file;
    static constexpr std::uint32_t addr  = F.addr32(S);
    static constexpr std::uint32_t words = Source::size == GprTransferSize::Bits128 ? 4u : 1u;

    Source source;
};

/**
 * @brief Associate a runtime value with a generated CFG field.
 */
template <const Field& F, Sec S>
inline constexpr FieldAssignment<F, S> set(const std::uint32_t value)
{
    return {value};
}

/**
 * @brief Associate a compile-time value with a generated CFG field.
 */
template <const Field& F, Sec S, std::uint32_t Value>
inline constexpr ConstantFieldAssignment<F, S, Value> set()
{
    return {};
}

namespace detail
{

/**
 * @brief A composed physical CFG word used only by the write backend.
 *
 * @p Mask is part of the type, allowing the Tensix backend to prune unused
 * RMWCIB byte writes at compile time.
 */
template <RegisterFile File, std::uint32_t Addr, std::uint32_t Mask>
class ConfigWord
{
public:
    static constexpr RegisterFile file  = File;
    static constexpr std::uint32_t addr = Addr;
    static constexpr std::uint32_t mask = Mask;

    std::uint32_t data;
};

/**
 * @brief A lazily encoded word containing exactly one field assignment.
 *
 * Keeping the unshifted value until emission lets a multi-word write compile
 * like the equivalent sequence of typed single-field writes. In particular,
 * it avoids extending encoded temporary lifetimes across later word updates.
 */
template <typename Assignment>
class SingleFieldWord
{
public:
    static constexpr RegisterFile file  = Assignment::file;
    static constexpr std::uint32_t addr = Assignment::addr;
    static constexpr std::uint32_t mask = Assignment::mask;

    Assignment assignment;
};

} // namespace detail

} // namespace hal::cfg
