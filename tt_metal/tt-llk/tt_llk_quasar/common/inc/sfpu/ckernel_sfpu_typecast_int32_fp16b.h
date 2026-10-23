// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{
// Calculates Typecast for number of rows of output SFPU ops (Quasar = 2 rows)
//
// Unpack-to-Dest copies Int32 L1 bits as two's-complement (see _mul_int32_), so the value is loaded
// as sfpi::vInt rather than sfpi::vSMag: convert<> then emits the 2SC → SM cast that the int → fp32
// cast mode requires, followed by the SM → fp32 and fp32 → fp16b round-nearest-even steps.
//
// An Int32 source also forces 32-bit Dest, so both accesses name their layout explicitly (I32 /
// F16b, the sfpmem INT32 / FP16B modes) rather than letting DataLayout::Default re-derive it from
// ALU_FORMAT_SPEC_REG / ACC_CTRL_SFPU_Fp32.
inline void _calculate_typecast_int32_to_fp16b_rows()
{
    const sfpi::vInt value = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();

    sfpi::dst_reg[0].mode<sfpi::DataLayout::F16b>() = sfpi::convert<sfpi::vFloat16b>(value, sfpi::RoundMode::NearestEven);
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        _calculate_typecast_int32_to_fp16b_rows();
        sfpi::dst_reg++; // increments by 2 rows (SFP_DESTREG_STRIDE), one SFPU pass
    }
}

} // namespace sfpu
} // namespace ckernel
