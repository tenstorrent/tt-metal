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
// `slot_idx` and `kv_actual_global` (tokens, tile-aligned) are read on-device by the writer
// kernel from `metadata` — a small uint32 DRAM tensor holding the runner's h2d_socket_sync payload
// in canonical layout [slot_id, actual_start, actual_end] (slot_idx = index 0, kv_actual_global =
// actual_start = index 1). They never touch the host dispatch path, so they stay out of the program
// hash and successive users/chunks reuse one cached program (per layer).
//
// Cache slot is addressed with users-outer, layers-inner linearization — the op composes
// `batch_idx = slot_idx * num_layers + layer_idx`.
//
// In-place: returns a handle to `cache`.
ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const ttnn::Tensor& metadata,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t cluster_axis);

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn {
using operations::experimental::deepseek_prefill::update_padded_kv_cache::update_padded_kv_cache;
}  // namespace ttnn
