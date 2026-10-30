// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sfpu_isinf_isnan_()
{
    // SFPU microcode
#if !(defined(MM_THROTTLE) && MM_THROTTLE > 0)
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    if constexpr (operation == SfpuType::isinf)
    {
        TTI_SFPENCC(3, 0, 12, 10);
        TTI_SFPLOADI(0, 10, 0x001c);
        TTI_SFPLOADI(0, 8, 0x8300);
        TTI_SFPCONFIG(0, 4, 0);
        TTI_SFPCONFIG(0x100, 8, 1);
        TTI_SFPLOADI(0, 0, 0x0000);
        TTI_SFPLOADI(2, 0, 0x7f80);
        TTI_SFPNOP;
        TTI_SFPENCC(3, 0, 0, 10);
        lltt::record(0, 5);
        TTI_SFPLOADMACRO((0 << 2) | 1, 0, ADDR_MOD_3, 0);
        TTI_SFPSETSGN(0, 1, 1, 1);
        TTI_SFPIADD(0, 2, 1, 6);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPSTORE(10, 0, ADDR_MOD_2, 0);
#pragma GCC unroll 0
        for (int d = 0; d + 0 < ITERATIONS; d += 1)
        {
            lltt::replay(0, 5);
        }
        TTI_SFPNOP;
        TTI_SFPCONFIG(0x000, 8, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        return;
    }
    else if constexpr (operation == SfpuType::isposinf)
    {
        TTI_SFPENCC(3, 0, 12, 10);
        TTI_SFPLOADI(0, 10, 0x0014);
        TTI_SFPLOADI(0, 8, 0x8300);
        TTI_SFPCONFIG(0, 4, 0);
        TTI_SFPCONFIG(0x100, 8, 1);
        TTI_SFPLOADI(0, 0, 0x0000);
        TTI_SFPLOADI(2, 0, 0x7f80);
        TTI_SFPNOP;
        TTI_SFPENCC(3, 0, 0, 10);
        lltt::record(0, 4);
        TTI_SFPLOADMACRO((0 << 2) | 1, 0, ADDR_MOD_3, 0);
        TTI_SFPIADD(0, 2, 1, 6);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPSTORE(10, 0, ADDR_MOD_2, 0);
#pragma GCC unroll 0
        for (int d = 0; d + 0 < ITERATIONS; d += 1)
        {
            lltt::replay(0, 4);
        }
        TTI_SFPNOP;
        TTI_SFPCONFIG(0x000, 8, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        return;
    }
    else if constexpr (operation == SfpuType::isneginf)
    {
        TTI_SFPENCC(3, 0, 12, 10);
        TTI_SFPLOADI(0, 10, 0x0014);
        TTI_SFPLOADI(0, 8, 0x8300);
        TTI_SFPCONFIG(0, 4, 0);
        TTI_SFPCONFIG(0x100, 8, 1);
        TTI_SFPLOADI(0, 0, 0x0000);
        TTI_SFPLOADI(2, 0, 0xff80);
        TTI_SFPNOP;
        TTI_SFPENCC(3, 0, 0, 10);
        lltt::record(0, 4);
        TTI_SFPLOADMACRO((0 << 2) | 1, 0, ADDR_MOD_3, 0);
        TTI_SFPIADD(0, 2, 1, 6);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPSTORE(10, 0, ADDR_MOD_2, 0);
#pragma GCC unroll 0
        for (int d = 0; d + 0 < ITERATIONS; d += 1)
        {
            lltt::replay(0, 4);
        }
        TTI_SFPNOP;
        TTI_SFPCONFIG(0x000, 8, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        return;
    }
    else if constexpr (operation == SfpuType::isnan)
    {
        TTI_SFPENCC(3, 0, 12, 10);
        TTI_SFPLOADI(0, 10, 0x0014);
        TTI_SFPLOADI(0, 8, 0x8300);
        TTI_SFPCONFIG(0, 4, 0);
        TTI_SFPCONFIG(0x100, 8, 1);
        TTI_SFPLOADI(0, 0, 0x0000);
        TTI_SFPLOADI(2, 0, 0x7f80);
        TTI_SFPNOP;
        TTI_SFPENCC(3, 0, 0, 10);
        lltt::record(0, 4);
        TTI_SFPLOADMACRO((0 << 2) | 1, 0, ADDR_MOD_3, 0);
        TTI_SFPSETSGN(0, 1, 1, 1);
        TTI_SFPIADD(0, 2, 1, 2);
        TTI_SFPSTORE(10, 0, ADDR_MOD_2, 0);
#pragma GCC unroll 0
        for (int d = 0; d + 0 < ITERATIONS; d += 1)
        {
            lltt::replay(0, 4);
        }
        TTI_SFPNOP;
        TTI_SFPCONFIG(0x000, 8, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        return;
    }
    else if constexpr (operation == SfpuType::isfinite)
    {
        TTI_SFPENCC(3, 0, 12, 10);
        TTI_SFPLOADI(0, 10, 0x0014);
        TTI_SFPLOADI(0, 8, 0x8300);
        TTI_SFPCONFIG(0, 4, 0);
        TTI_SFPCONFIG(0x100, 8, 1);
        TTI_SFPLOADI(0, 0, 0x0000);
        TTI_SFPLOADI(2, 2, 0xffff);
        TTI_SFPLOADI(2, 8, 0x7f7f);
        TTI_SFPNOP;
        TTI_SFPENCC(3, 0, 0, 10);
        lltt::record(0, 4);
        TTI_SFPLOADMACRO((0 << 2) | 1, 0, ADDR_MOD_3, 0);
        TTI_SFPSETSGN(0, 1, 1, 1);
        TTI_SFPIADD(0, 2, 1, 10);
        TTI_SFPSTORE(10, 0, ADDR_MOD_2, 0);
#pragma GCC unroll 0
        for (int d = 0; d + 0 < ITERATIONS; d += 1)
        {
            lltt::replay(0, 4);
        }
        TTI_SFPNOP;
        TTI_SFPCONFIG(0x000, 8, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        return;
    }
#endif
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in  = sfpi::dst_reg[0];
        sfpi::vFloat res = 0.0f;

        if constexpr (operation == SfpuType::isinf)
        {
            v_if (sfpi::is_inf(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == SfpuType::isposinf)
        {
            v_if (sfpi::is_pos(in) && sfpi::is_inf(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == SfpuType::isneginf)
        {
            v_if (sfpi::is_neg(in) && sfpi::is_inf(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == SfpuType::isnan)
        {
            v_if (sfpi::is_nan(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == SfpuType::isfinite)
        {
            v_if (sfpi::is_finite(in))
            {
                res = 1.0f;
            }
            v_endif;
        }

        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
