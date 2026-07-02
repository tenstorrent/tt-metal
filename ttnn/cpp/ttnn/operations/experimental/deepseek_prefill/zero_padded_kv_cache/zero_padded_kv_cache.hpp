// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache {

// Zeroes the migration pad window [valid_global, ceil_pad_align(valid_global)) of a block-cyclic KV
// cache, in place. Both the single-shot fill and chunked update_padded_kv_cache write full 32-row
// tiles, so the tokens between the last real token (valid_global) and the next pad_align boundary
// hold stale data; this op clears them so the decode side reads clean zeros. The window is up to
// pad_align-1 tokens (1-4 tiles) and may straddle a chip boundary -- each chip zeroes its own share,
// derived from valid_global + chunk_size_global + the device's coordinate along cluster_axis.
//
// The boundary (partial) tile is read, multiplied by an in-kernel row-mask and written back; the
// fully-pad tiles are written from the L1 zeros buffer.
//
// Cache slot is addressed users-outer, layers-inner: batch_idx = slot_idx * num_layers + layer_idx.
// valid_global and slot_idx stay out of the program hash, so successive chunks reuse one cached
// program. In-place: returns a handle to `cache`. Two call forms (identical results):

// (1) Scalar form: per-call slot_idx/valid_global are host values (common runtime args, patched on
//     cache hits).
ttnn::Tensor zero_padded_kv_cache(
    const ttnn::Tensor& cache,
    uint32_t slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t valid_global,
    uint32_t chunk_size_global,
    uint32_t cluster_axis,
    uint32_t pad_align = 128);

// (2) Tensor form (traceable): the reader/writer read slot_idx and valid_global (= actual_end)
//     on-device, each from its own 1-element uint32 DRAM tensor (replicated across the mesh). These are
//     the un-packed per-element views of the runner's h2d_socket_sync payload [slot_id, actual_start,
//     actual_end] (slot_idx = element 0, valid_global = element 2). Off the host dispatch path, so one
//     captured program replays across chunks.
ttnn::Tensor zero_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& slot_idx,
    const ttnn::Tensor& valid_global,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t chunk_size_global,
    uint32_t cluster_axis,
    uint32_t pad_align = 128);

}  // namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache

namespace ttnn {
using operations::experimental::deepseek_prefill::zero_padded_kv_cache::zero_padded_kv_cache;
}  // namespace ttnn
