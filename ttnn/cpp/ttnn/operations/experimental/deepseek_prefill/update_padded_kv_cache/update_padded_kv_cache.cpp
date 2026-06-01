// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "update_padded_kv_cache.hpp"
#include "device/update_padded_kv_cache_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const ttnn::Tensor& metadata,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t cluster_axis) {
    return ttnn::prim::update_padded_kv_cache(cache, input, metadata, layer_idx, num_layers, cluster_axis);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache
