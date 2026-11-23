// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

// The read mirror of update_padded_kv_cache: each chip copies back the page-rows the write stored for the chunk that
// starts at `kv_actual_global` -- the same per-chip offset, from the same block-cyclic math -- for layers
// [first_layer, first_layer + num_read_layers) of user `slot_idx`, into a [1, 1, num_read_layers * chunk_local,
// width] tensor (layer-major; chunk_local = chunk_size_global / (sp * tp) rows per chip). Writing those rows back with
// update_padded_kv_cache(cluster_axis=None, input_row_offset=l * chunk_local, input_rows=chunk_local) at the same
// slot / kv_actual_global / valid_global reproduces the cache bit for bit on a mesh of the same shape, which is how
// a pipeline stage forwards a chunk's stored KV rows to a downstream copy of the cache without the TP-replicated
// fields the write consumed.
//
// `slot_idx` and `kv_actual_global` are 1-element uint32 DRAM tensors read on device, so the read is trace-safe.
// `cluster_axis` / `tp_axis` are the linearization the cache was written with. Rows past the cache end are not
// read (the write never stores them); they are left unwritten in `output`. ROW_MAJOR caches only. An FP8_E4M3 cache
// is read as UINT8 rows of the same width (its bytes), which update_padded_kv_cache accepts for an FP8 cache.
ttnn::Tensor read_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& slot_idx,
    const ttnn::Tensor& kv_actual_global,
    uint32_t first_layer,
    uint32_t num_read_layers,
    uint32_t num_layers,
    uint32_t chunk_size_global,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> tp_axis = std::nullopt,
    const std::optional<ttnn::Tensor>& output = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn {
using operations::experimental::deepseek_prefill::update_padded_kv_cache::read_padded_kv_cache;
}  // namespace ttnn
