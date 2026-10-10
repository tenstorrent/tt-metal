// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_bf16_reciprocal_complement_core.h"
#include "ckernel_sfpu_bf16_sfpi_isa.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16
{
template <class Config>
inline void init_reciprocal_complement()
{
    sfpu_reciprocal_init<false>();
    sfpi::vConstFloatPrgm1 = Config::kCoefficients[Config::kDegree];
    sfpi::vConstFloatPrgm2 = Config::kCoefficients[Config::kDegree - 1];
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    // The body's load macro: its two instruction templates, its sequence and its misc settings.
    TTI_SFPSETSGN(0, p_sfpu::LREG2, 15, 0);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG0, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(0, sfpi::SFPLOADI_MOD0_UPPER, Config::kMacroSequenceBits >> 16);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(0, sfpi::SFPLOADI_MOD0_LOWER, Config::kMacroSequenceBits & 0xffffu);
    ::ckernel::sfpu::bf16_sfpi::sfpconfig(0, 4, 0);
    ::ckernel::sfpu::bf16_sfpi::sfpconfig(0xf00, 8, 1);
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
}
} // namespace ckernel::sfpu::bf16
