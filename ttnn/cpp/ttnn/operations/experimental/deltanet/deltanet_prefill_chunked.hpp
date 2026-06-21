// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <optional>
#include <vector>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
namespace ttnn::experimental {
std::vector<Tensor> deltanet_prefill_chunked(
    const Tensor& k, const Tensor& q, const Tensor& v, const Tensor& z,
    const Tensor& Kdec, const Tensor& KiT, const Tensor& Qd,
    const Tensor& dcol, const Tensor& betacol, const Tensor& dlast,
    const Tensor& recurrent_state, const Tensor& norm_weight,
    uint32_t num_heads, uint32_t k_head_dim, uint32_t v_head_dim,
    uint32_t chunk, uint32_t n_chunks, uint32_t seq_len,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn::experimental
