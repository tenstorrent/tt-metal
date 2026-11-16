// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attention_qkv_decode_nanobind.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/attention_qkv_decode/attention_qkv_decode.hpp"

namespace ttnn::operations::experimental::deepseek::detail {

void bind_attention_qkv_decode(::nanobind::module_& mod) {
    namespace nb = nanobind;
    ttnn::bind_function<"attention_qkv_decode", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        DeepSeek V4.1 decode attention up to SDPA, for one token, as one program per chip: q_a and kv
        projections with their RMSNorms, q_b, RoPE of q and kv, and the kv row written into the ring.

            q_a = rms_norm(tokens @ w_q_a) * q_a_gamma
            q   = rope(q_a @ w_q_b)               per head, on its trailing rope_dim channels
            kv  = rope(rms_norm(tokens @ w_kv) * kv_gamma)
            kv_cache[cache_pos] = kv

        Blackhole only, on the 12x10 grid: 40 q_a cores (x 8..11), 16 kv cores (x 0..7, y 8..9) and 64
        q_b cores (x 0..7, y 0..7).

        Args:
            tokens (ttnn.Tensor): ``[1, K]`` bfloat16 ROW_MAJOR HEIGHT_SHARDED replica
                (``all_gather_for_matmul``) covering the q_a and kv cores.
            w_q_a (ttnn.Tensor): ``[K, q_lora]`` TILE DRAM, ND-sharded one ``[K, 32]`` slab per q_a core.
            w_kv (ttnn.Tensor): ``[K, head_dim]`` TILE DRAM, ND-sharded one ``[K, 32]`` slab per kv core.
            w_q_b (ttnn.Tensor): ``[q_lora, heads * head_dim]`` TILE DRAM, ND-sharded one
                ``[q_lora, N / 64]`` slab per q_b core.
            q_a_gamma (ttnn.Tensor): ``[1, q_lora]`` bfloat16 1x32 TILE, WIDTH_SHARDED one tile per q_a core.
            kv_gamma (ttnn.Tensor): ``[1, head_dim]`` bfloat16 1x32 TILE, WIDTH_SHARDED one tile per kv core.
            cos (ttnn.Tensor): ``[1, 1, 1, rope_dim]`` bfloat16 TILE, interleaved.
            sin (ttnn.Tensor): ``[1, 1, 1, rope_dim]`` bfloat16 TILE, interleaved.
            kv_cache (ttnn.Tensor): ``[.., rows, head_dim]`` bfloat16 TILE, interleaved; row ``cache_pos``
                is overwritten in place.
            cache_pos (ttnn.Tensor): int32 ``[1]`` ROW_MAJOR, read on device.

        Keyword Args:
            epsilon (float): RMSNorm epsilon of both norms.
            memory_config (Optional[ttnn.MemoryConfig]): q's memory config, default every head
                HEIGHT_SHARDED on core (0, 0), flash_mqa_decode's layout.
            prefetcher_pipes (List[ttnn.experimental.PrefetcherPipe]): every pipe of the q_a, kv and q_b
                tensor-prefetcher rings, which then deliver the weights' slabs instead of DRAM reads: one
                whole-slab entry per core, so each call drains one queued request per weight. The receivers
                must be exactly the roles' cores. Empty (the default) reads the DRAM slabs.
            w_comp (Optional[ttnn.Tensor]): ``[K, N_c]`` TILE DRAM, ND-sharded one ``[K, 32]`` slab per output
                tile, at most 64 tiles: extra output columns ``tokens @ w_comp`` (a column-split compressor),
                computed on that many q_b cores ahead of their q_b work. Always read from DRAM.
            comp_output (Optional[ttnn.Tensor]): ``[1, 1, 1, N_c]`` bfloat16 ROW_MAJOR DRAM, overwritten with
                ``tokens @ w_comp``. Given with ``w_comp``.

        Several-row blocks (DSpark verify): ``tokens`` a bfloat16 TILE with ``(h, 32)`` tiles, h in 2, 4, 8,
        HEIGHT_SHARDED one ``[h, K]`` block per core. Each row is its own token (per-row RMSNorms and RoPE), the
        weights read once. ``cos`` / ``sin`` are then ``(h, 32)`` TILE ``[1, 1, h, rope_dim]`` (row j = block row
        j's angles); ``kv_cache`` is the staging tensor, 32x32 TILE ``[1, 1, 32, head_dim]``, whose rows 0..h-1
        receive the kv rows (the ring is not written and ``cache_pos`` is not read); ``comp_output`` is ``(h, 32)``
        TILE ``[1, 1, h, N_c]``; q is ``[1, 1, heads * h, head_dim]`` head-major (head hd's row j at hd * h + j).
        ``commit_rows`` (optional, blocks only): ROW_MAJOR bfloat16 ``[1, 1, h, head_dim]``, L1 HEIGHT_SHARDED on
        one core (shard ``[h, head_dim]``), also receives the kv rows: ``paged_update_cache``'s input for the ring.

        Returns:
            ttnn.Tensor: q ``[1, 1, heads, head_dim]`` bfloat16 ROW_MAJOR (``[1, 1, heads * h, head_dim]`` for a
            block).
        )doc",
        &ttnn::experimental::deepseek::attention_qkv_decode,
        nb::arg("tokens"),
        nb::arg("w_q_a"),
        nb::arg("w_kv"),
        nb::arg("w_q_b"),
        nb::arg("q_a_gamma"),
        nb::arg("kv_gamma"),
        nb::arg("cos"),
        nb::arg("sin"),
        nb::arg("kv_cache"),
        nb::arg("cache_pos"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-6f,
        nb::arg("memory_config") = std::nullopt,
        nb::arg("prefetcher_pipes") = ttnn::PrefetcherPipeList{},
        nb::arg("w_comp") = std::nullopt,
        nb::arg("comp_output") = std::nullopt,
        nb::arg("commit_rows") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
