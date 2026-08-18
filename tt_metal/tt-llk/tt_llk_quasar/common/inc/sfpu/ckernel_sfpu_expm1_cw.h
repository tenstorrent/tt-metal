// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_compat.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_polyval.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Shared Cody-Waite expm1 helper used by ELU, CELU and SELU.  This is
// intentionally the same SFPI algorithm and API as Blackhole; it contains no
// architecture-specific instruction or destination-layout dependency.
inline constexpr float CW_INV_LN2    = 1.4426950408889634f;
inline constexpr float CW_NEG_LN2_HI = -0.6931152343750000f;
inline constexpr float CW_NEG_LN2_LO = -3.19461832987e-05f;

sfpi_inline sfpi::vFloat expm1_cw_clamped(sfpi::vFloat x)
{
    x = compat::fp_max(x, -87.0f);

    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);
    sfpi::vFloat tmp        = x * CW_INV_LN2 + c231;
    sfpi::vFloat k_f        = tmp - c231;
    sfpi::vFloat r          = k_f * CW_NEG_LN2_HI + x;
    r                       = r + k_f * CW_NEG_LN2_LO;

#ifdef INP_FLOAT32
    sfpi::vFloat h = PolynomialEvaluator::eval(r, 1.0f, 5.0000000000e-01f, 1.6666504741e-01f, 4.1666239500e-02f, 8.3691505715e-03f, 1.3948583510e-03f);
#else
    sfpi::vFloat h = PolynomialEvaluator::eval(r, 1.0f, 4.9999371171e-01f, 1.6666433215e-01f, 4.1875664145e-02f, 8.3751315251e-03f);
#endif
    h = r * h;

    constexpr int kC231Bias = 0x4B3FFF81;
    sfpi::vFloat two_k      = sfpi::setexp(1.0f, sfpi::as<sfpi::vInt>(tmp) - kC231Bias);
    return (two_k - 1.0f) + two_k * h;
}

} // namespace ckernel::sfpu
