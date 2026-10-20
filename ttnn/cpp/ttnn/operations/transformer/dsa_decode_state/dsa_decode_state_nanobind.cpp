// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "dsa_decode_state_nanobind.hpp"
#include "dsa_decode_state.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>

namespace ttnn::operations::transformer {

void bind_dsa_decode_state(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused DSA (DeepSeek sparse attention) decode state maintenance, one program. From the decode stem matmul
        output ``stem`` [1, 1, B, Ws] (bf16 TILE; column blocks q_a | kv_a | indexer k | pool gate at the given
        offsets) it computes q_resid = rmsnorm(q_a) * w (multicore, the return value) and, one core per user b at
        position a = positions[b]: the latent row rmsnorm(kv_a) * w written at row a % 32 of page
        latent_page_table[b, a // 32] of ``latent_cache``; the LayerNorm'd k row and the raw gate row written at
        slot a % P of the user's ``ring_k`` / ``ring_g`` block; and the pooled key of pool a // P,
        sum_j softmax_j(g_ring[j] + ape[j]) k_ring[j] over the P ring rows, written at row (a // P) % 32 of block
        b * pool_blocks + (a // P) // 32 of ``pool_cache``. All four caches are updated IN PLACE.

        Args:
            stem [1,1,B,Ws] bf16 TILE; latent_cache [n_blocks,1,32,L] bf16 TILE; latent_page_table [B, blocks_per_seq]
            int32 ROW_MAJOR; positions [B] int32 ROW_MAJOR; ring_k, ring_g [>=B,1,32,D] bf16 TILE (block = user);
            pool_cache [B*pool_blocks,1,32,D] bf16 TILE (identity page table); qa_norm_w [1,1,1,Lq], kva_norm_w
            [1,1,1,L], k_norm_w / k_norm_b [1,1,1,D] fp32 TILE; ape [1,1,32,D] fp32 TILE (rows < P: the in-pool
            position embedding, rows >= P: -1e9); selectors [33*32, 32] bf16 TILE (tile r < 32: 1 at (0, r), tile 32:
            all ones); the column offsets and widths (tile multiples; L, Lq multiples of 128, D = 128).

        Keyword Args:
            eps_rms (q_a / kv_a RMSNorm, default 1e-5), eps_ln (k LayerNorm, default 1e-6), memory_config,
            compute_kernel_config (HiFi4 / fp32 acc by default; fp32 acc required).

        Returns:
            ttnn.Tensor: q_resid [1, 1, B, Lq] bf16 TILE. The in-pool softmax is shifted by the first ring row (shift
            invariant; in-pool logit spreads beyond ~80 would overflow). Needs B + ceil(B / 32) cores.
        )doc";

    ttnn::bind_function<"dsa_decode_state", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::dsa_decode_state,
        nb::arg("stem").noconvert(),
        nb::arg("latent_cache").noconvert(),
        nb::arg("latent_page_table").noconvert(),
        nb::arg("positions").noconvert(),
        nb::arg("ring_k").noconvert(),
        nb::arg("ring_g").noconvert(),
        nb::arg("pool_cache").noconvert(),
        nb::arg("qa_norm_w").noconvert(),
        nb::arg("kva_norm_w").noconvert(),
        nb::arg("k_norm_w").noconvert(),
        nb::arg("k_norm_b").noconvert(),
        nb::arg("ape").noconvert(),
        nb::arg("selectors").noconvert(),
        nb::arg("qa_offset"),
        nb::arg("kva_offset"),
        nb::arg("k_offset"),
        nb::arg("g_offset"),
        nb::arg("q_lora_rank"),
        nb::arg("latent_dim"),
        nb::arg("index_dim"),
        nb::arg("pool_size"),
        nb::arg("pool_blocks"),
        nb::kw_only(),
        nb::arg("eps_rms") = 1e-5f,
        nb::arg("eps_ln") = 1e-6f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
