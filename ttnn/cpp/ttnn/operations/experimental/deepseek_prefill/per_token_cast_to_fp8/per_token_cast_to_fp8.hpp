// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8 {

inline constexpr uint32_t BLOCK_W = 128;
inline constexpr float E4M3_MAX_NORMAL = 448.0f;
inline constexpr float SCALE_CLAMP_MIN = 1.0e-4f;  // DeepEP clamps amax to >= 1e-4 before /448

std::tuple<ttnn::Tensor, ttnn::Tensor> per_token_cast_to_fp8(
    const ttnn::Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    bool round_scale_to_power_of_two = false);

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8

namespace ttnn {
using operations::experimental::deepseek_prefill::per_token_cast_to_fp8::per_token_cast_to_fp8;
}  // namespace ttnn
