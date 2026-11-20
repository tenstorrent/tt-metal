// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>

namespace ttnn::operations::moreh::moreh_layer_norm {

inline uint32_t tile_mean_factor_bits(float scaler, bool lastdim, uint32_t H, uint32_t W) {
    const double s = std::bit_cast<float>(std::bit_cast<uint32_t>(scaler) & 0xFFFF0000u);
    const double n0 = static_cast<double>(std::min(W, 32u)) * (lastdim ? 1.0 : static_cast<double>(std::min(H, 32u)));
    return std::bit_cast<uint32_t>(static_cast<float>(1.0 / (n0 * (lastdim ? s : s * s))));
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm
