// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "csa_compressor.hpp"

#include "device/csa_compressor_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/compressor_state_exchange/compressor_state_exchange.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::csa_compressor {

std::tuple<Tensor, Tensor, Tensor> csa_compressor(
    const Tensor& kv,
    const Tensor& gate,
    const Tensor& position_bias,
    const Tensor& initial_kv_state,
    const Tensor& initial_score_state,
    uint32_t seq_len_actual,
    uint32_t first_token_position,
    uint32_t cluster_axis,
    ::ttnn::ccl::Topology topology) {
    auto local_state = ttnn::prim::csa_prepare_state(
        kv,
        gate,
        position_bias,
        initial_kv_state,
        initial_score_state,
        seq_len_actual,
        first_token_position,
        cluster_axis);

    auto [predecessor_kv, predecessor_score] = compressor_state_exchange::compressor_state_exchange(
        local_state[0], local_state[1], initial_kv_state, initial_score_state, cluster_axis, topology);

    auto result = ttnn::prim::csa_compress(
        kv, gate, position_bias, predecessor_kv, predecessor_score, seq_len_actual, first_token_position, cluster_axis);
    return {result[0], result[1], result[2]};
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::csa_compressor
