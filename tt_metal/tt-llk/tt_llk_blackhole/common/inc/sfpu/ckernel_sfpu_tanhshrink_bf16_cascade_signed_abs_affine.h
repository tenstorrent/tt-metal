// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_tanhshrink_bf16_signed_abs_affine.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::tanhshrink
{
template <typename Config, int Iterations = 8>
inline void calculate_signed_abs_affine()
{
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    // Stock tanhshrink's call reads nothing its init sets, so each row loads its own upper coefficients
    // and rounds and stores itself; ADDR_MOD_7 holds DEST and INCRWC steps it.
    ttpoly::tanhshrink::signed_abs_affine_pin_coefficients<Config>();
#pragma GCC unroll 1
    for (int row = 0; row < Iterations; ++row)
    {
        ttpoly::tanhshrink::signed_abs_affine_held_body<Config, ADDR_MOD_7>();
        TTI_INCRWC(0, 2, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16::tanhshrink
