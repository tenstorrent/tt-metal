// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "zero_padded_kv_cache.hpp"

#include <optional>

#include "device/zero_padded_kv_cache_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache {

// Scalar form: no metadata tensor; slot_idx/valid_global are host scalars.
ttnn::Tensor zero_padded_kv_cache(
    const ttnn::Tensor& cache,
    uint32_t slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t valid_global,
    uint32_t chunk_size_global,
    uint32_t cluster_axis,
    uint32_t pad_align) {
    return ttnn::prim::zero_padded_kv_cache(
        cache,
        /*metadata=*/std::nullopt,
        slot_idx,
        layer_idx,
        num_layers,
        valid_global,
        chunk_size_global,
        cluster_axis,
        pad_align);
}

// Metadata form: slot_idx/valid_global read on-device from `metadata` (the scalar attrs are unused).
ttnn::Tensor zero_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& metadata,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t chunk_size_global,
    uint32_t cluster_axis,
    uint32_t pad_align) {
    return ttnn::prim::zero_padded_kv_cache(
        cache,
        metadata,
        /*slot_idx=*/0,
        layer_idx,
        num_layers,
        /*valid_global=*/0,
        chunk_size_global,
        cluster_axis,
        pad_align);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache
