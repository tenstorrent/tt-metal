// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/prefetcher_pipe.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// DeepSeek V4.1 decode attention up to SDPA, for one token, as one program per chip:
//
//   q_a = rms_norm(tokens @ w_q_a) * q_a_gamma
//   q   = rope(q_a @ w_q_b)               per head, on its trailing rope_dim channels
//   kv  = rope(rms_norm(tokens @ w_kv) * kv_gamma)
//   kv_cache[cache_pos] = kv
//
// Args:
//   tokens:    ``[1, K]`` bfloat16 ROW_MAJOR HEIGHT_SHARDED replica (all_gather_for_matmul) on the whole
//              12x10 grid.
//   w_q_a:     ``[K, q_lora]`` TILE DRAM, ND-sharded one ``[K, 32]`` slab per q_a core.
//   w_kv:      ``[K, head_dim]`` TILE DRAM, ND-sharded one ``[K, 32]`` slab per kv core.
//   w_q_b:     ``[q_lora, heads * head_dim]`` TILE DRAM, ND-sharded one ``[q_lora, N / 64]`` slab per q_b core.
//   q_a_gamma: ``[1, q_lora]`` bfloat16 1x32 TILE, WIDTH_SHARDED one tile per q_a core.
//   kv_gamma:  ``[1, head_dim]`` bfloat16 1x32 TILE, WIDTH_SHARDED one tile per kv core.
//   cos, sin:  ``[1, 1, 1, rope_dim]`` bfloat16 TILE, interleaved (decode's RoPE tables).
//   kv_cache:  ``[.., rows, head_dim]`` bfloat16 TILE, interleaved; row cache_pos is overwritten.
//   cache_pos: int32 ``[1]`` ROW_MAJOR, read on device.
//   prefetcher_pipes: optional tensor-prefetcher pipes (every pipe of the q_a, kv and q_b rings) that deliver
//              the three weights' slabs instead of DRAM reads, one whole-slab entry per core per call. The
//              receivers must be exactly the roles' cores. Each call drains one request per weight, queued in the
//              weights' own ND-sharded layout.
//   w_comp:    optional ``[K, N_c]`` TILE DRAM, ND-sharded one ``[K, 32]`` slab per output tile (N_c / 32 <= 64):
//              extra output columns ``tokens @ w_comp`` (e.g. a column-split compressor), computed on that many
//              of the q_b cores before their q_b work. Always read from DRAM.
//   comp_output: ``[1, 1, 1, N_c]`` bfloat16 ROW_MAJOR DRAM, overwritten with ``tokens @ w_comp``; given with
//              w_comp.
//
// Returns: q ``[1, 1, heads, head_dim]`` bfloat16 ROW_MAJOR, by default flash_mqa_decode's layout (every
// head HEIGHT_SHARDED on core (0, 0)).
Tensor attention_qkv_decode(
    const Tensor& tokens,
    const Tensor& w_q_a,
    const Tensor& w_kv,
    const Tensor& w_q_b,
    const Tensor& q_a_gamma,
    const Tensor& kv_gamma,
    const Tensor& cos,
    const Tensor& sin,
    const Tensor& kv_cache,
    const Tensor& cache_pos,
    float epsilon = 1e-6f,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const ttnn::PrefetcherPipeList& prefetcher_pipes = {},
    const std::optional<Tensor>& w_comp = std::nullopt,
    const std::optional<Tensor>& comp_output = std::nullopt);

}  // namespace ttnn::experimental::deepseek
