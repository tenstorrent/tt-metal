// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "api/compute/compute_kernel_api.h"
#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {

template <std::uint32_t valid, std::uint32_t addr = 0>
inline void softmax_pad_neg_inf_cols_from(sfpi::vInt local_col) {
    constexpr int face_col = static_cast<int>(((addr / 16) % 2) * 16);
    constexpr int parity = (addr & 2) ? 1 : 0;
    constexpr int in_face = static_cast<int>(valid) - face_col;
    if constexpr (in_face <= 0) {
        sfpi::dst_reg[addr / 2] = -std::numeric_limits<float>::infinity();
    } else if constexpr (in_face < 16) {
        constexpr int keep = in_face > parity ? (in_face - parity + 1) / 2 : 0;
        v_if(local_col >= 2 * keep) { sfpi::dst_reg[addr / 2] = -std::numeric_limits<float>::infinity(); }
        v_endif;
    }
    if constexpr (addr < 62) {
        softmax_pad_neg_inf_cols_from<valid, addr + 2>(local_col);
    }
}

template <std::uint32_t valid>
inline void softmax_pad_neg_inf_cols() {
    softmax_pad_neg_inf_cols_from<valid>(sfpi::vConstTileId & 0xE);
}

}  // namespace ckernel::sfpu
#endif

ALWI void softmax_pad_neg_inf_init() { MATH(SFPU_UNARY_INIT(unused)); }

template <std::uint32_t valid>
ALWI void softmax_pad_neg_inf_tile(std::uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, softmax_pad_neg_inf_cols, (valid), idst, VectorMode::None));
}
