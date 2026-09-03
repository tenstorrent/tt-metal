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
 * thread CFG and RMWCIB/WRCFG/RDCFG for state CFG. TensixScalarUnit emits
 * REG2FLOP for GPR-backed THCON writes.
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
 * @brief Width of a GPR-backed WRCFG or REG2FLOP transfer.
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
 * that WRCFG wrote; one NOP of separation is enough. Deferred is for a sequence
 * that already provides it. It is a compile-time policy and introduces no
 * runtime control flow.
 */
enum class WrcfgCompletion : std::uint8_t
{
    Wait,
    Deferred
};

} // namespace hal::cfg
