// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <tuple>

#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::fp8_quant_common {

inline constexpr uint32_t BLOCK_W = 128;
inline constexpr float E4M3_MAX_NORMAL = 448.0f;
inline constexpr float SCALE_CLAMP_MIN = 1.0e-4f;  // DeepEP clamps amax to >= 1e-4 before /448

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
    TT_FATAL(H % BLOCK_W == 0, "Per-token cast: hidden dim H={} must be a multiple of BLOCK_W={}", H, BLOCK_W);
    ttsl::SmallVector<uint32_t> dims;
    dims.reserve(rank);
    for (size_t i = 0; i + 1 < rank; ++i) {
        dims.push_back(static_cast<uint32_t>(input_shape[i]));
    }
    dims.push_back(H / BLOCK_W);
    return ttnn::Shape(std::move(dims));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::fp8_quant_common
