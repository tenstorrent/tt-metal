// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "zero_padded_kv_cache.hpp"

#include <optional>

#include "device/zero_padded_kv_cache_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache {

// Scalar form: no per-element tensors; slot_idx/valid_global are host scalars.
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
        /*slot_idx_tensor=*/std::nullopt,
        /*valid_global_tensor=*/std::nullopt,
        slot_idx,
        layer_idx,
        num_layers,
        valid_global,
        chunk_size_global,
        cluster_axis,
        pad_align);
}

// Tensor form: slot_idx/valid_global read on-device from two 1-element uint32 tensors (the scalar
// attrs are unused).
ttnn::Tensor zero_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& slot_idx,
    const ttnn::Tensor& valid_global,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t chunk_size_global,
    uint32_t cluster_axis,
    uint32_t pad_align) {
    return ttnn::prim::zero_padded_kv_cache(
        cache,
        slot_idx,
        valid_global,
        /*slot_idx=*/0,
        layer_idx,
        num_layers,
        /*valid_global=*/0,
        chunk_size_global,
        cluster_axis,
        pad_align);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache
