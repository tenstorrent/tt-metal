// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// Replay slot 0 holds one LOADMACRO per iteration.
inline constexpr std::uint32_t _exp_loadmacro_replay_len_(const int num_sfpu_iterations)
{
    return static_cast<std::uint32_t>(num_sfpu_iterations);
}

// Program the exp LOADMACRO sequence and record one macro per iteration into replay slot 0.
// Each macro is self-contained: LREG <- LD[addr]; STG <- EXP[LREG]; ST[addr + STORE_OFFSET] <- STG.
// The final macro sets the `done` bit to reset the SrcS dvalids, so callers must not clear them.
// load_sfpmem / store_sfpmem are sfpmem format codes resolved by the caller via _sfpu_sfpmem_type_
// (Float16 needs explicit FP16A; DEFAULT never resolves to it).
// STORE_OFFSET must be a compile-time constant so the store is captured via the immediate field.
template <std::uint32_t STORE_OFFSET>
inline void _exp_init_loadmacro_(
    const std::uint32_t load_base_addr, const int num_sfpu_iterations, const std::uint32_t load_sfpmem, const std::uint32_t store_sfpmem)
{
    LLK_ASSERT(num_sfpu_iterations <= 4, "Replay cycles LREG0-3 (d & 3); >4 in-flight macros would reuse a live LREG");

    // LOADMACRO CONTROL: DEFAULT_STORE_INSMOD = store_sfpmem. With
    // STORE_INHERITS_INSMOD=0 this register, not the captured SFPSTORE, sets the store format.
    TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, store_sfpmem);
    TTI_SFPCONFIG(0x0000, p_sfpconfig::MACRO_CTRL, 0x0);
    TTI_SFPNOP(0, 0, 0); // SFPCONFIG hazard: no instr may issue the cycle after SFPCONFIG

    // Instr reg 4: STG <- EXP[LREG]  (captured via the MACRO_CAPTURE backdoor, not executed)
    TTI_SFPNONLINEAR(p_sfpu::LREG0 /* VC */, p_sfpu::MACRO_CAPTURE_INSTR4 /* VD */, p_sfpnonlinear::EXP_MODE);

    // Instr reg 6: ST[load_addr + STORE_OFFSET] <- STG (the capture index also selects the staging register as store source)
    TT_SFPSTORE(p_sfpu::MACRO_CAPTURE_INSTR6, store_sfpmem, ADDR_MOD_0, 0b0, STORE_OFFSET);

    // Sequence register 0:
    //   SIMPLE = 0x44 -> instr 4 (EXP) with USE_STAGING (result to STG)
    //   STORE  = 0xCE -> STORE slot enabled, STORE_ADDR_OFFSET set, instr 6 (store from STG)
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0244); // [MAD | SIMPLE]
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0xCE02); // [STORE | ROUND]
    TTI_SFPCONFIG(0x0000, p_sfpconfig::MACRO_SEQ0, 0x0);
    TTI_SFPNOP(0, 0, 0); // SFPCONFIG hazard: no instr may issue the cycle after SFPCONFIG

    load_replay_buf(
        0,
        _exp_loadmacro_replay_len_(num_sfpu_iterations),
        false,
        0,
        0,
        [load_base_addr, num_sfpu_iterations, load_sfpmem]
        {
            for (int d = 0; d < num_sfpu_iterations; d++)
            {
                const std::uint32_t done = (d == num_sfpu_iterations - 1);
                // addr is the [10:1] field, hence >> 1
                TT_SFPLOADMACRO(0, d & 3, load_sfpmem, ADDR_MOD_1, done, (load_base_addr + (d << 1)) >> 1, 0);
            }
        });
}

// Op-word that executes the LOADMACRO replay recorded by `_exp_init_loadmacro_`.
// Replay slot 0 and length stay private to this header; use with MOP / programmed paths.
inline std::uint32_t _exp_loadmacro_op_(const int num_sfpu_iterations)
{
    return TT_OP_REPLAY(0, _exp_loadmacro_replay_len_(num_sfpu_iterations), 0, 0, 0, 0);
}

} // namespace sfpu
} // namespace ckernel
