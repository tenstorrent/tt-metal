// SPDX-License-Identifier: Apache-2.0
#include "deltanet_prefill_chunked.hpp"
#include "device/deltanet_prefill_chunked_device_operation.hpp"
namespace ttnn::experimental {
std::vector<Tensor> deltanet_prefill_chunked(
    const Tensor& k, const Tensor& q, const Tensor& v, const Tensor& z,
    const Tensor& Kdec, const Tensor& KiT, const Tensor& Qd,
    const Tensor& dcol, const Tensor& betacol, const Tensor& dlast,
    const Tensor& recurrent_state, const Tensor& norm_weight,
    uint32_t num_heads, uint32_t k_head_dim, uint32_t v_head_dim,
    uint32_t chunk, uint32_t n_chunks, uint32_t seq_len,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::deltanet_prefill_chunked(
        k, q, v, z, Kdec, KiT, Qd, dcol, betacol, dlast, recurrent_state, norm_weight,
        num_heads, k_head_dim, v_head_dim, chunk, n_chunks, seq_len, memory_config);
}
}  // namespace ttnn::experimental
