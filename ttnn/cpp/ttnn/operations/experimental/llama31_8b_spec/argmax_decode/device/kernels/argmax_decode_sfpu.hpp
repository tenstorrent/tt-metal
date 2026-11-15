// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void argmax_decode_step(const uint32_t tile, const uint32_t first) {
    constexpr uint32_t dst_tile_size_sfpi = 32;
    const sfpi::vInt low = static_cast<int>(0x00800000u + (0x3FFFu - tile));
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vUInt b = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::dst_reg[0]));
        sfpi::vInt hi = sfpi::as<sfpi::vInt>(b) & 0xFFFF;
        sfpi::vInt m = hi & 0x7FFF;
        sfpi::vInt order = m + 0x8000;
        v_if(hi >= 0x8000) { order = 0x8000 - m; }
        v_endif;
        v_if(m > 0x7F80) { order = 0x10000; }
        v_endif;
        sfpi::vInt key = (order << 14) + low;
        if (first) {
            sfpi::dst_reg[dst_tile_size_sfpi] = sfpi::as<sfpi::vFloat>(key);
        } else {
            sfpi::vInt best = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[dst_tile_size_sfpi]));
            v_if(key > best) { sfpi::dst_reg[dst_tile_size_sfpi] = sfpi::as<sfpi::vFloat>(key); }
            v_endif;
        }
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
