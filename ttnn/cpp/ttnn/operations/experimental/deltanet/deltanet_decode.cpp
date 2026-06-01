// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_decode.hpp"
#include "device/deltanet_decode_device_operation.hpp"

namespace ttnn::experimental {

std::vector<Tensor> deltanet_decode(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& decay,
    const Tensor& beta,
    const Tensor& state,
    uint32_t num_heads,
    uint32_t k_head_dim,
    uint32_t v_head_dim,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::deltanet_decode(
        query, key, value, decay, beta, state,
        num_heads, k_head_dim, v_head_dim, memory_config);
}

}  // namespace ttnn::experimental
