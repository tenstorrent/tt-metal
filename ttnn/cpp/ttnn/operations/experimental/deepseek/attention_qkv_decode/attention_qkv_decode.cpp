// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attention_qkv_decode.hpp"

#include "device/attention_qkv_decode_device_operation.hpp"

namespace ttnn::experimental::deepseek {

Tensor attention_qkv_decode(
    const Tensor& tokens,
    const Tensor& w_q_a,
    const Tensor& w_kv,
    const Tensor& w_q_b,
    const Tensor& q_a_gamma,
    const Tensor& kv_gamma,
    const Tensor& cos,
    const Tensor& sin,
    const Tensor& kv_cache,
    const Tensor& cache_pos,
    float epsilon,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::PrefetcherPipeList& prefetcher_pipes,
    const std::optional<Tensor>& w_comp,
    const std::optional<Tensor>& comp_output) {
    return ttnn::prim::attention_qkv_decode(
        tokens,
        w_q_a,
        w_kv,
        w_q_b,
        q_a_gamma,
        kv_gamma,
        cos,
        sin,
        kv_cache,
        cache_pos,
        epsilon,
        memory_config,
        prefetcher_pipes,
        w_comp,
        comp_output);
}

}  // namespace ttnn::experimental::deepseek
