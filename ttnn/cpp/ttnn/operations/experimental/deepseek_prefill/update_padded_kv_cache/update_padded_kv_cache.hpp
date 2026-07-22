// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

// Writes a `chunk_local`-row input slab into a KV cache at a per-device start offset, derived
// from a single global token count `kv_actual_global` and the device's coordinate along
// `cluster_axis`. When `kv_actual_global` aligns to a chunk boundary every device writes at the
// same local offset (the chunked-natural case); when it does not, devices around the boundary
// write at different offsets so that new tokens overwrite the trailing pad cells of the prior
// cache before spilling into the next slab.
//
// Cache slot is addressed with users-outer, layers-inner linearization — the op composes
// `batch_idx = slot_idx * num_layers + layer_idx`. `slot_idx` and `kv_actual_global` (tokens,
// tile-aligned) stay out of the program hash either way, so successive users/chunks reuse one cached
// program (per layer).
//
// Supports TILE and ROW_MAJOR layouts (the op is a pure page copy; the per-chip offset math is
// expressed in 32-row-aligned page units, identical for both). `cache` and `input` must share
// layout and dtype: block-float (bfloat8_b/bfloat4_b) is TILE-only; FP8_E4M3 is ROW_MAJOR-only.
//
// In-place: returns a handle to `cache`. Two call forms (identical results):

// (1) Scalar form (original): per-call `slot_idx`/`kv_actual_global` are host values, passed as
//     common runtime args and patched on cache hits.
ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    uint32_t slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t kv_actual_global,
    uint32_t cluster_axis);

// (2) Per-element-tensor form (traceable): `slot_idx`/`kv_actual_global` are read on-device by the
//     writer kernel from two 1-element uint32 DRAM tensors ([1,1,1,1], ROW_MAJOR, replicated across
//     the mesh) — `slot_idx` holds the user slot, `kv_actual_global` holds the prior valid global KV
//     length in tokens (tile-aligned). The writer reads element [0] of each. Because they never touch
//     the host dispatch path, this form is trace-safe (one captured program replays across
//     chunks/users).
ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const ttnn::Tensor& slot_idx,
    const ttnn::Tensor& kv_actual_global,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t cluster_axis);

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn {
using operations::experimental::deepseek_prefill::update_padded_kv_cache::update_padded_kv_cache;
}  // namespace ttnn
