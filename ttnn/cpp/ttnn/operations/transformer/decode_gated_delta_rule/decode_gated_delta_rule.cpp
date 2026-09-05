// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_gated_delta_rule.hpp"

#include <cmath>

#include "device/decode_gated_delta_rule_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor> decode_gated_delta_rule(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& g,
    std::optional<float> scale,
    const std::optional<ttnn::Tensor>& initial_state,
    bool inplace_state,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    const auto& qs = q.logical_shape();  // [B,1,H,K]
    const uint32_t K = qs[3];
    const float s = scale.value_or(1.0f / std::sqrt(static_cast<float>(K)));

    // No host preprocessing: the T=1 [B,1,H,*] shapes are consumed directly
    // (the reader gathers each head's row out of the shared TILE pages), so
    // the whole graph above is one device program.
    auto results = ttnn::prim::decode_gated_delta_rule(
        q, k, v, beta, g, initial_state, inplace_state, s, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
    return {results[0], results[1]};
}

}  // namespace ttnn::transformer
