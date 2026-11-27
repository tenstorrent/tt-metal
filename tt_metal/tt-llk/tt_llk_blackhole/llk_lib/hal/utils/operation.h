// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

namespace hal
{

class Operation;

namespace detail
{
constexpr Operation make_operation(std::uint32_t word);
} // namespace detail

/**
 * @brief One Tensix instruction encoded by a HAL encoder, to issue later with @ref run.
 *
 * Only HAL encoders construct an Operation, so a kernel cannot issue a raw instruction word. It is one
 * 32-bit word, so a constexpr table of Operations selected at run time costs what a table of raw words does.
 * It converts to std::uint32_t for MOP programs and comparisons.
 *
 * @code
 * constexpr hal::Operation table[] = {
 *     hal::replay::get_operation(hal::replay::BufferRange {4, 6}),
 *     hal::mop::Runner<hal::mop::MopTemplate::Template0> {.count = 1, .mask = 0x1}.operation()};
 * hal::run(table[index]);
 * @endcode
 */
class Operation
{
public:
    constexpr operator std::uint32_t() const
    {
        return word_;
    }

private:
    constexpr explicit Operation(const std::uint32_t word) : word_(word)
    {
    }

    std::uint32_t word_;

    friend constexpr Operation detail::make_operation(std::uint32_t word);
    friend void run(Operation operation);
};

namespace detail
{
/** @brief Wrap a word a HAL encoder produced; only HAL encoders may call this. */
constexpr Operation make_operation(const std::uint32_t word)
{
    return Operation {word};
}

/** @brief Issue a compile-time word as an immediate instruction. */
template <std::uint32_t Word>
inline __attribute__((always_inline)) void run_immediate()
{
    TTI_INSN(Word);
}
} // namespace detail

/**
 * @brief Issue an encoded operation.
 *
 * A word known at compile time issues as an immediate instruction; a runtime one is written to the
 * instruction buffer.
 */
inline __attribute__((always_inline)) void run(const Operation operation)
{
    const std::uint32_t word = operation.word_;
    if (__builtin_constant_p(word))
    {
        TTI_INSN(word);
    }
    else
    {
        TT_INSN(word);
    }
}

} // namespace hal
