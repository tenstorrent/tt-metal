// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"

namespace hal::misc
{

// ---------------- AutoSync ----------------

/**
 * @brief Describe an ahead-of-time resource declaration for one AutoSync instruction class (RESOURCEDECL).
 *
 * Declared resources are persistent and replace the hardware's per-class defaults (the
 * default table lives in instructions/assembly.yaml). MOP- and replay-expanded instructions
 * inherit the parent MOP/REPLAY instruction's declared class.
 *
 * @note No LLK call site exists; the hardware defaults are in use everywhere today.
 */
struct ResourceDeclaration
{
    std::uint8_t instruction_class = 0; // AutoSync class, 4 bits
    std::uint16_t resources        = 0; // 9-bit resource mask: {TDMA RW, GPR RW, CFG zones, CFG RW}
    std::uint16_t linger_time      = 1; // cycles the declaration lingers after issue

    /** @brief Encode the operation. */
    constexpr std::uint32_t get_operation() const;
};

// ---------------- TDMA engine ----------------

/** @brief Select which TDMA component a flush drains. */
enum class FlushScope : std::uint8_t
{
    All              = 0x0,
    ThreadController = 0x1,
    Unpacker0        = 0x2,
    Unpacker1        = 0x4,
    Packer           = 0x8
};

constexpr FlushScope operator|(const FlushScope lhs, const FlushScope rhs)
{
    return static_cast<FlushScope>(static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs));
}

/**
 * @brief Encode a blocking drain of the TDMA engine or a subset of it (FLUSHDMA).
 *
 * @tparam Scope: Component selection; @ref FlushScope::All drains the entire engine.
 * @note Blocks every thread's ThCon issue while draining; the ISA recommends a targeted
 *       STALLWAIT instead — see @ref hal::sync::wait. No LLK call site exists.
 */
template <FlushScope Scope = FlushScope::All>
constexpr std::uint32_t flush_tdma_operation()
{
    static_assert(static_cast<std::uint32_t>(Scope) < 16u, "TDMA flush selection is 4 bits");
    return TT_OP_FLUSHDMA(static_cast<std::uint32_t>(Scope));
}

/** @brief Drain the TDMA engine or a subset of it, blocking until idle (FLUSHDMA). */
template <FlushScope Scope = FlushScope::All>
inline __attribute__((always_inline)) void flush_tdma()
{
    INSTRUCTION_WORD((flush_tdma_operation<Scope>()));
}

/**
 * @brief Encode a TDMA soft reset (RSTDMA).
 *
 * @note The instruction metadata carries "fixme; remove instr?" and no ISA documentation
 *       exists; no LLK call site exists. Wrapped for encoding completeness only.
 */
constexpr std::uint32_t reset_tdma_operation()
{
    return TT_OP_RSTDMA;
}

/** @brief Soft-reset the TDMA engine (RSTDMA). */
inline __attribute__((always_inline)) void reset_tdma()
{
    INSTRUCTION_WORD(reset_tdma_operation());
}

/**
 * @brief Encode the reserved TBUFCMD instruction.
 *
 * @note Reserved for future use with no documented behavior; wrapped for encoding
 *       completeness only.
 */
constexpr std::uint32_t tbuf_command_operation()
{
    return TT_OP_TBUFCMD;
}

/** @brief Issue the reserved TBUFCMD instruction. */
inline __attribute__((always_inline)) void tbuf_command()
{
    INSTRUCTION_WORD(tbuf_command_operation());
}

// ---------------- Implementation ----------------

namespace detail
{

constexpr bool is_valid(const ResourceDeclaration declaration)
{
    return declaration.instruction_class < 16u && declaration.resources < (1u << 9) && declaration.linger_time < (1u << 11);
}

constexpr void reject_invalid_constant(const bool valid)
{
    if (__builtin_is_constant_evaluated() && !valid)
    {
        __builtin_trap();
    }
}

#ifdef ENABLE_LLK_ASSERT
inline __attribute__((always_inline)) void assert_valid(const ResourceDeclaration declaration)
{
    LLK_ASSERT(declaration.instruction_class < 16u, "RESOURCEDECL instruction class is 4 bits");
    LLK_ASSERT(declaration.resources < (1u << 9), "RESOURCEDECL resource mask is 9 bits");
    LLK_ASSERT(declaration.linger_time < (1u << 11), "RESOURCEDECL linger time is 11 bits");
}
#endif

} // namespace detail

/** @brief Return whether a resource declaration is encodable. */
constexpr bool is_valid(const ResourceDeclaration declaration)
{
    return detail::is_valid(declaration);
}

inline constexpr __attribute__((always_inline)) std::uint32_t ResourceDeclaration::get_operation() const
{
    detail::reject_invalid_constant(detail::is_valid(*this));
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(*this);
    }
#endif
    return TT_OP_RESOURCEDECL(linger_time, resources, instruction_class);
}

/** @brief Issue a compile-time descriptor as one immediate instruction. */
// Compile-time path
template <auto Operation>
inline __attribute__((always_inline)) void run()
{
    static_assert(detail::is_valid(Operation), "invalid descriptor");
    constexpr std::uint32_t operation = Operation.get_operation();
    INSTRUCTION_WORD(operation);
}

/** @brief Issue a runtime-selected descriptor. */
// Runtime path
template <typename Operation>
inline __attribute__((always_inline)) void run(const Operation operation)
{
    ckernel::instrn_buffer[0] = operation.get_operation();
}

} // namespace hal::misc
