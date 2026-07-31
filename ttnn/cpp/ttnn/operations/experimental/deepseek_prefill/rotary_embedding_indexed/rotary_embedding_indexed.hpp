// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed {

// KV-pad-aware indexed RoPE for chunked prefill. Applies rotary embedding to a per-chip input
// chunk, indexing into SP-sharded cos/sin caches at a per-device offset derived on-device from a
// single global valid-KV length (`kv_actual_global`) and the device's coordinate along
// `cluster_axis`.
//
// The cos/sin caches must be sharded across `cluster_axis` in block-cyclic order keyed by the
// per-device chunk size (so device c's shard holds, in contiguous local-row order, the rope values
// for every global position it will carry). The op then derives the chunk's start row in that
// shard the same way the per-chip kv-cache writer derives its `update_idxt`, so the boundary chip's
// older-then-wrap token layout is read with a single contiguous offset.
//
// `kv_actual_global` (tokens, tile-aligned) stays out of the program hash, so successive chunks reuse
// one cached program. Returns a new tensor with the same spec as `input`. Two call forms (identical
// results):

// (1) Scalar form: `kv_actual_global` is a host scalar held in a common runtime arg, patched on cache
//     hits.
ttnn::Tensor rotary_embedding_indexed(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t kv_actual_global,
    uint32_t cluster_axis,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

// (2) Tensor form (traceable): `kv_actual_global` is its OWN 1-element uint32 DRAM tensor that the reader
//     reads on-device (element [0]). Off the host dispatch path, so one captured program replays across
//     chunks (the host updates the 1-element tensor in place per chunk).
ttnn::Tensor rotary_embedding_indexed(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    const ttnn::Tensor& kv_actual_global,
    uint32_t cluster_axis,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed

namespace ttnn {
using operations::experimental::deepseek_prefill::rotary_embedding_indexed::rotary_embedding_indexed;
}  // namespace ttnn
