// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common.h"
#include "llk_math_matmul.h"

using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

static_assert(ELTWISE_MATH_ROWS == 4, "4row_quasar overrides require ELTWISE_MATH_ROWS == 4");

/*************************************************************************
 * LLK MATH MATMUL CUSTOM NO MOP
 *************************************************************************/

template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_run_no_mop_(const bool reuse_a)
{
    constexpr std::uint32_t FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 1 : to_underlying(MATH_FIDELITY_TYPE);

    if constexpr (ENABLE_2X_FORMAT)
    {
        constexpr std::uint32_t replay_buf_len = _llk_math_matmul_di_replay_buf_len_4row_<true>();

        for (std::uint32_t phase = 0; phase < FIDELITY_PHASES - 1; phase++)
        {
            TTI_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
            TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x7, 0x4, ADDR_MOD_1, 0xF);
        }

        TTI_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
        if (reuse_a)
        {
            TTI_MVMULDI(p_setrwc::CLR_A, 0x0, 0x7, 0x4, ADDR_MOD_2, 0xF);
        }
        else
        {
            TTI_MVMULDI(p_setrwc::CLR_B, 0x0, 0x7, 0x4, ADDR_MOD_2, 0xF);
        }
    }
    else
    {
        constexpr std::uint32_t replay_buf_len = _llk_math_matmul_replay_buf_len_<false>();

        for (std::uint32_t phase = 0; phase < FIDELITY_PHASES - 1; phase++)
        {
            TTI_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);
        }

        TTI_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
        if (reuse_a)
        {
            TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0);
        }
        else
        {
            TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_5, 0);
        }
    }
}

template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_init_no_mop_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    if constexpr (ENABLE_2X_FORMAT)
    {
        _llk_math_matmul_di_addrmod_<MATH_FIDELITY_TYPE>(ct_dim, rt_dim);
        _llk_math_matmul_di_load_replay_4row_<true>();
    }
    else
    {
        _llk_math_matmul_addrmod_<MATH_FIDELITY_TYPE, false>(ct_dim, rt_dim);
        _llk_math_matmul_load_replay_<false>();
    }

    _reset_counters_<p_setrwc::SET_ABD_F>();
}

template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_block_no_mop_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(0);

    const bool reuse_a          = ct_dim >= rt_dim;
    const std::uint32_t t_dim   = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim;

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++)
        {
            _llk_math_matmul_run_no_mop_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(reuse_a);

            if (rut == (rut_dim - 1))
            {
                if (reuse_a)
                {
                    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_AB_F);
                }
                else
                {
                    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_AB_F);
                }
            }
        }

        if (!reuse_a && ct_dim >= 2)
        {
            TT_SETRWC(p_setrwc::CLR_NONE, 0, 64 * (t + 1), p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::C_TO_CR_MODE, 0, p_setrwc::SET_D);
        }
    }
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
