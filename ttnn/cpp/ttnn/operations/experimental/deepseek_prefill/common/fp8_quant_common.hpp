// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::fp8_quant_common {

inline constexpr uint32_t SCALE_GROUP_SIZE = 128;
inline constexpr uint32_t SCALE_GROUP_TILES = 4;  // SCALE_GROUP_SIZE / TILE_WIDTH (=32)
inline constexpr float E4M3_MAX_NORMAL = 448.0f;

inline std::tuple<uint32_t, uint32_t> infer_M_H(const ttnn::Shape& shape) {
    const auto rank = shape.size();
    TT_FATAL(rank >= 2, "Per-token cast ops require rank >= 2 input, got rank {}", rank);
    uint32_t M = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        M *= static_cast<uint32_t>(shape[i]);
    }
    const uint32_t H = static_cast<uint32_t>(shape[rank - 1]);
    return {M, H};
}

inline ttnn::Shape scale_shape_from_input(const ttnn::Shape& input_shape) {
    const auto rank = input_shape.size();
    const uint32_t H = static_cast<uint32_t>(input_shape[rank - 1]);
    TT_FATAL(
        H % SCALE_GROUP_SIZE == 0,
        "Per-token cast: hidden dim H={} must be a multiple of SCALE_GROUP_SIZE={}",
        H,
        SCALE_GROUP_SIZE);
    ttnn::SmallVector<uint32_t> dims;
    dims.reserve(rank);
    for (size_t i = 0; i + 1 < rank; ++i) {
        dims.push_back(static_cast<uint32_t>(input_shape[i]));
    }
    dims.push_back(H / SCALE_GROUP_SIZE);
    return ttnn::Shape(std::move(dims));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::fp8_quant_common
