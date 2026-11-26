// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "unpack.h"
#include "utils/helpers.h"

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
    TTI_INSN(thread_operation());
}

/** @brief Encode a ThCon/TDMA queue bubble (DMANOP). */
constexpr std::uint32_t scalar_operation()
{
    return TT_OP_DMANOP;
}

/** @brief Consume one TDMA instruction slot and cycle (DMANOP). */
inline __attribute__((always_inline)) void scalar()
{
    TTI_INSN(scalar_operation());
}

/** @brief Encode an SFPU issue bubble (SFPNOP). */
constexpr std::uint32_t vector_operation()
{
    return static_cast<std::uint32_t>(TT_OP_SFPNOP);
}

/** @brief Consume one SFPU issue slot and cycle (SFPNOP). */
inline __attribute__((always_inline)) void vector()
{
    TTI_INSN(vector_operation());
}

/**
 * @brief Encode a pure-delay unpacker bubble (UNPACR_NOP, delay flavor).
 *
 * @tparam Engine: Unpacker whose queue the bubble occupies.
 * @note The source-bank clear flavor of UNPACR_NOP, with its optional data-valid handoff, is an
 *       unpacker operation, not a NOP: use @ref hal::unpack::Unpacker::clear.
 */
template <hal::unpack::Engine Engine>
constexpr std::uint32_t unpacker_operation()
{
    static_assert(hal::unpack::detail::is_valid(Engine), "UNPACR_NOP engine must be Unpacker0 or Unpacker1");
    return TT_OP_UNPACR_NOP(
        hal::to_underlying(Engine),
        0 /*Stream_Id*/,
        0 /*Msg_Clr_Cnt*/,
        0 /*Set_Dvalid*/,
        0 /*Clr_to1_fmt_Ctrl*/,
        0 /*Stall_Clr_Cntrl*/,
        0 /*Bank_Clr_Ctrl*/,
        0 /*Src_ClrVal_Ctrl*/,
        hal::to_underlying(hal::unpack::detail::NopMode::Delay));
}

/** @brief Consume one unpacker instruction slot and cycle (UNPACR_NOP, delay flavor). */
template <hal::unpack::Engine Engine>
inline __attribute__((always_inline)) void unpacker()
{
    TTI_INSN((unpacker_operation<Engine>()));
}

} // namespace hal::nop
