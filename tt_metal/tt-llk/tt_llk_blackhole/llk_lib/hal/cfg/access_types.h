// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace hal::cfg
{

/**
 * @brief How a configuration access reaches hardware.
 *
 * MMIO uses the TRISC/BRISC memory-mapped path. TensixCfgUnit emits SETC16 for
 * thread CFG and RMWCIB/WRCFG/RDCFG for state CFG. TensixScalarUnit is unsupported
 * because Blackhole does not handle REG2FLOP properly. Use TensixCfgUnit (WRCFG)
 * for GPR-backed writes.
 */
enum class Access : std::uint8_t
{
    MMIO,
    TensixCfgUnit,
    TensixScalarUnit
};

/**
 * @brief Thread-CFG bank selected by a TRISC/BRISC MMIO read.
 *
 * Current selects the issuing TRISC's private bank and is therefore valid
 * only in a TRISC build. T0/T1/T2 explicitly select a bank and can also be
 * used by BRISC.
 */
enum class ThreadTarget : std::uint8_t
{
    Current,
    T0,
    T1,
    T2
};

/**
 * @brief Width of a GPR-backed WRCFG transfer.
 */
enum class GprTransferSize : std::uint8_t
{
    Bits32,
    Bits128
};

/**
 * @brief Whether a WRCFG helper emits its completion NOP immediately.
 *
 * The instruction immediately after a WRCFG must not consume the configuration
 * that WRCFG wrote; one NOP of separation is enough. Deferred, the default,
 * emits no NOP; request Wait on the transfer whose result the next instruction
 * consumes. It is a compile-time policy and introduces no runtime control flow.
 */
enum class WrcfgCompletion : std::uint8_t
{
    Wait,
    Deferred
};

/**
 * @brief Scratch::val section that @ref add_scratch adds to its destination word.
 *
 * Thread selects the issuing thread's section: section 0 for unpack, 1 for math, 2 for pack.
 */
enum class ScratchSec : std::uint8_t
{
    S0     = 0,
    S1     = 1,
    S2     = 2,
    Thread = 3
};

} // namespace hal::cfg
