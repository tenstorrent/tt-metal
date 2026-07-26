// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

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
// `batch_idx = slot_idx * num_layers + layer_idx`. For a single-user prefill workload, callers
// pass `slot_idx = 0` and the desired `layer_idx`.
//
// `slot_idx` and `kv_actual_global` are per-call scalars held in common runtime args and patched on
// cache hits, so their values stay out of the program hash and successive users/chunks reuse one
// cached program (per layer).
//
// Supports TILE and ROW_MAJOR layouts (the op is a pure page copy; the per-chip offset math is
// expressed in 32-row-aligned page units, identical for both). `cache` and `input` must share
// layout and dtype: block-float (bfloat8_b/bfloat4_b) is TILE-only; FP8_E4M3 is ROW_MAJOR-only.
//
// In-place: returns a handle to `cache`.
ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    uint32_t slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t kv_actual_global,
    uint32_t cluster_axis);

// Paged variant for 5120-token prefill bundles. `cache_pool` keeps the existing cache-compatible
// per-device shape [num_physical_bundles * num_layers, 1, 5120 / SP, head_dim], with bundle-major,
// layer-inner batch linearization. `page_table` is a persistent ROW_MAJOR INT32 device tensor
// [num_slots, max_logical_bundles] mapping each global logical bundle to a physical bundle ID.
// Each bundle's 32-token subpages are deterministic. Page-table contents, slot_idx and
// kv_actual_global are runtime values and do not create new programs.
ttnn::Tensor paged_update_padded_kv_cache(
    const ttnn::Tensor& cache_pool,
    const ttnn::Tensor& input,
    const ttnn::Tensor& page_table,
    uint32_t slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t kv_actual_global,
    uint32_t cluster_axis);

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn {
using operations::experimental::deepseek_prefill::update_padded_kv_cache::paged_update_padded_kv_cache;
using operations::experimental::deepseek_prefill::update_padded_kv_cache::update_padded_kv_cache;
}  // namespace ttnn
