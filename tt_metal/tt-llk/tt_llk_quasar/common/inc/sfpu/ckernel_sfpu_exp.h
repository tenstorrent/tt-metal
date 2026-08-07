// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{

// replay slot 0 = num_sfpu_iterations self-contained LOADMACROs (each does load -> EXP->STG -> store)
inline constexpr std::uint32_t _exp_loadmacro_replay_len_(const int num_sfpu_iterations)
{
    return static_cast<std::uint32_t>(num_sfpu_iterations);
}

// Program a self-contained exp LOADMACRO sequence and record one macro per element into replay
// slot 0. Each macro does: LREG <- LD[addr]; STG <- EXP[LREG]; ST[addr + store_offset] <- STG.
// The store is folded into the macro's STORE slot via the STORE_ADDR_OFFSET flag, so the result
// lands in a separate SrcS slice (store_offset from the load base) with no discrete SFPSTORE.
// The final macro sets the `done` bit to reset the SrcS dvalids, so callers must NOT issue
// _llk_math_eltwise_sfpu_srcs_clear_vlds_ or drain NOPs per slice.
// SFPLOADMACRO addr is the [10:1] field, hence >> 1; STORE_OFFSET is a raw address delta and must
// be a compile-time constant so the instr-reg-6 store is captured via the immediate (TTI_) path.
template <std::uint32_t STORE_OFFSET>
inline void _exp_init_loadmacro_(const std::uint32_t load_base_addr, const int num_sfpu_iterations)
{
    // LOADMACRO CONTROL register (config_dest 0x8): default store mode, zero all other fields.
    TTI_SFPLOADI(0x0, 0x2, 0x0002);
    TTI_SFPCONFIG(0x0000, 0x8, 0x0);
    TTI_SFPNOP(0, 0, 0); // SFPCONFIG hazard: no instr may issue the cycle after SFPCONFIG (state affects S2)

    // Instr reg 4: STG <- EXP[LREG]  (VD=0xC is the capture backdoor)
    TTI_SFPNONLINEAR(0x0 /* VC */, 0xC /* VD */, p_sfpnonlinear::EXP_MODE);

    // Instr reg 6: ST[load_addr + STORE_OFFSET] <- STG  (0xE = staging register source)
    TTI_SFPSTORE(0xE, 0x2, 0x0, 0b0, STORE_OFFSET);

    // Sequence register 0:
    //   SIMPLE = 0x44 -> instr 4 (EXP) with USE_STAGING (result to STG)
    //   STORE  = 0xCE -> STORE slot enabled, STORE_ADDR_OFFSET set, instr 6 (store from STG)
    TTI_SFPLOADI(0x0, 0xA, 0x0244); // [MAD | SIMPLE]
    TTI_SFPLOADI(0x0, 0x8, 0xCE02); // [STORE | ROUND]
    TTI_SFPCONFIG(0x0000, 0x4, 0x0);
    TTI_SFPNOP(0, 0, 0); // SFPCONFIG hazard: no instr may issue the cycle after SFPCONFIG (state affects S2)

    load_replay_buf(
        0,
        _exp_loadmacro_replay_len_(num_sfpu_iterations),
        false,
        0,
        0,
        [load_base_addr, num_sfpu_iterations]
        {
            for (int d = 0; d < num_sfpu_iterations; d++)
            {
                const std::uint32_t done = (d == num_sfpu_iterations - 1) ? 1 : 0;
                TT_SFPLOADMACRO(0, d & 3, 0x2, 0x1, done, (load_base_addr + (d << 1)) >> 1, 0);
            }
        });
}

} // namespace sfpu
} // namespace ckernel
