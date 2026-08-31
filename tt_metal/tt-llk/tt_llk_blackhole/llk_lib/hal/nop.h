// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "unpack.h"

namespace hal::nop
{

// A NOP is only meaningful relative to the queue it occupies; one function per pipeline.

/** @brief Encode a one-cycle frontend issue bubble (NOP). */
constexpr std::uint32_t thread_operation()
{
    return TT_OP_NOP;
}

/** @brief Consume one frontend issue slot and cycle (NOP). */
inline __attribute__((always_inline)) void thread()
{
    INSTRUCTION_WORD(thread_operation());
}

/** @brief Encode a ThCon/TDMA queue bubble (DMANOP). */
constexpr std::uint32_t scalar_operation()
{
    return TT_OP_DMANOP;
}

/** @brief Consume one TDMA instruction slot and cycle (DMANOP). */
inline __attribute__((always_inline)) void scalar()
{
    INSTRUCTION_WORD(scalar_operation());
}

/** @brief Encode an SFPU issue bubble (SFPNOP). */
constexpr std::uint32_t vector_operation()
{
    return static_cast<std::uint32_t>(TT_OP_SFPNOP);
}

/** @brief Consume one SFPU issue slot and cycle (SFPNOP). */
inline __attribute__((always_inline)) void vector()
{
    INSTRUCTION_WORD(vector_operation());
}

/**
 * @brief Encode a pure-delay unpacker bubble (UNPACR_NOP, delay flavor).
 *
 * @tparam Engine: Unpacker whose queue the bubble occupies.
 * @note The side-effectful UNPACR_NOP flavors (source-bank fill, data-valid publish, stream
 *       pop) are unpacker operations, not NOPs, and belong to @ref hal::unpack.
 */
template <hal::unpack::Engine Engine>
constexpr std::uint32_t unpacker_operation()
{
    return TT_OP_UNPACR_NOP(static_cast<std::uint32_t>(Engine), 0, 0, 0, 0, 0, 0, 0, ckernel::p_unpacr_nop::UNP_NOP);
}

/** @brief Consume one unpacker instruction slot and cycle (UNPACR_NOP, delay flavor). */
template <hal::unpack::Engine Engine>
inline __attribute__((always_inline)) void unpacker()
{
    INSTRUCTION_WORD((unpacker_operation<Engine>()));
}

} // namespace hal::nop
