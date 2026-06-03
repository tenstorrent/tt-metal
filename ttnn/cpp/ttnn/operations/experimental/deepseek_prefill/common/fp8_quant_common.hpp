// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <tuple>

#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::fp8_quant_common {

inline constexpr uint32_t SCALE_GROUP_SIZE = 128;
inline constexpr uint32_t SCALE_GROUP_TILES = 4;  // SCALE_GROUP_SIZE / TILE_WIDTH (=32)
inline constexpr float E4M3_MAX_NORMAL = 448.0f;
inline constexpr float SCALE_CLAMP_MIN = 1.0e-4f;  // DeepEP clamps amax to >= 1e-4 before /448
// The LLK kernels process the width in 1024-element column-blocks (32 tiles). H need NOT be a
// multiple of this: the last column-block may be partial (fewer whole 128-groups) and the kernels
// zero-pad it. H must still be a multiple of SCALE_GROUP_SIZE (128) so groups are always full.
inline constexpr uint32_t COL_BLOCK_ELEMS = 1024;

inline std::tuple<uint32_t, uint32_t> infer_M_H(const ttnn::Shape& shape) {
    const auto rank = shape.size();
    TT_FATAL(rank >= 2, "Per-token cast ops require rank >= 2 input, got rank {}", rank);
    uint64_t M = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        M *= static_cast<uint64_t>(shape[i]);
        TT_FATAL(
            M <= std::numeric_limits<uint32_t>::max(),
            "Per-token cast: folded row count M={} exceeds uint32_t range",
            M);
    }
    TT_FATAL(
        static_cast<uint64_t>(shape[rank - 1]) <= std::numeric_limits<uint32_t>::max(),
        "Per-token cast: hidden dim H={} exceeds uint32_t range",
        shape[rank - 1]);
    const uint32_t H = static_cast<uint32_t>(shape[rank - 1]);
    return {static_cast<uint32_t>(M), H};
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
