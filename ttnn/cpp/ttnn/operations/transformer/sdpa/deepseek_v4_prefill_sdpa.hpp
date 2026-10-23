// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <cstdint>
#include <optional>

namespace ttnn::transformer {

// DeepSeek V4 prefill attention over two KV sources, Blackhole single-chip, no mask tensor.
//   q          [1, H, S, K_DIM] bf16 ROW_MAJOR, any H
//   window_kv  [1, 1, window_size + S, K_DIM] bf16 ROW_MAJOR: the window_size tail rows carried from the
//              previous chunk, then the chunk's own S rows. K == V.
//   start      1-element uint32/int32 ROW_MAJOR device tensor: the absolute position of q's row 0. Read on
//              device, so a captured trace replays at a new position.
//   cache_kv   optional [1, 1, T, K_DIM] bf16 ROW_MAJOR compressed KV cache. K == V.
//   cache_idx  optional [1, 1, S, TOPK] uint32 ROW_MAJOR: rows of cache_kv each query attends. 0xFFFFFFFF skips
//              a slot and may sit anywhere in the row; a row may be all sentinels. Given iff cache_kv is.
// Query s attends window_kv rows s + 1 .. s + window_size (the last one is the token itself), except rows
// holding positions before 0, plus its valid cache_idx rows; both sources share one softmax.
// compress_ratio  > 0: cache row w pools tokens [w * ratio, (w + 1) * ratio), so query s sees it only once
//                 w < (start + s + 1) / ratio; other listed rows are skipped like sentinels. 0: no such check.
// empty_tail      the window_size tail rows hold no token (the window restarts at the chunk), whatever start is.
// attention_sink: optional [1, 1, 1, H] bf16, in the scaled-logit domain as in sparse_sdpa (pass sink / scale).
// Returns [1, H, S, v_dim] ROW_MAJOR bf16, V being the leading v_dim columns of each K row.
// Preconditions not checked per element: every valid cache_idx entry is < T.
ttnn::Tensor deepseek_v4_prefill_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& window_kv,
    const ttnn::Tensor& start,
    const std::optional<ttnn::Tensor>& cache_kv = std::nullopt,
    const std::optional<ttnn::Tensor>& cache_idx = std::nullopt,
    uint32_t window_size = 128,
    std::optional<uint32_t> v_dim = std::nullopt,
    std::optional<float> scale = std::nullopt,
    uint32_t k_chunk_size = 128,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& attention_sink = std::nullopt,
    uint32_t compress_ratio = 0,
    bool empty_tail = false);

}  // namespace ttnn::transformer
