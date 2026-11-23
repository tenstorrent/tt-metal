// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "read_padded_kv_cache.hpp"

#include "device/read_padded_kv_cache_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

ttnn::Tensor read_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& slot_idx,
    const ttnn::Tensor& kv_actual_global,
    uint32_t first_layer,
    uint32_t num_read_layers,
    uint32_t num_layers,
    uint32_t chunk_size_global,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> tp_axis,
    const std::optional<ttnn::Tensor>& output) {
    return ttnn::prim::read_padded_kv_cache(
        cache,
        slot_idx,
        kv_actual_global,
        first_layer,
        num_read_layers,
        num_layers,
        chunk_size_global,
        cluster_axis,
        tp_axis,
        output);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache
