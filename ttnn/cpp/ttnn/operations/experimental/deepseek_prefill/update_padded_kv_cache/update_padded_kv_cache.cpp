// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "update_padded_kv_cache.hpp"

#include <optional>

#include "device/update_padded_kv_cache_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

// Scalar form: no metadata tensor; slot_idx/kv_actual_global are host scalars.
ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    uint32_t slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t kv_actual_global,
    uint32_t cluster_axis) {
    return ttnn::prim::update_padded_kv_cache(
        cache, input, /*metadata=*/std::nullopt, slot_idx, kv_actual_global, layer_idx, num_layers, cluster_axis);
}

// Metadata form: slot_idx/kv_actual_global read on-device from `metadata` (the scalar attrs are unused).
ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const ttnn::Tensor& metadata,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t cluster_axis) {
    return ttnn::prim::update_padded_kv_cache(
        cache, input, metadata, /*slot_idx=*/0, /*kv_actual_global=*/0, layer_idx, num_layers, cluster_axis);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache
