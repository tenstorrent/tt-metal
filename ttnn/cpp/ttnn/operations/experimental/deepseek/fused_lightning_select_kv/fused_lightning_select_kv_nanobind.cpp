// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_lightning_select_kv_nanobind.hpp"

#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/fused_lightning_select_kv/fused_lightning_select_kv.hpp"

namespace ttnn::operations::experimental::deepseek::fused_lightning_select_kv::detail {

void bind_fused_lightning_select_kv(nb::module_& mod) {
    ttnn::bind_function<"fused_lightning_select_kv", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Fused lightning indexer select for DeepSeek V4-Flash CSA.

        Scores the indexer key cache, keeps the top-k positions, and gathers those
        rows from the paged attention KV cache::

            scores  = indexer_score_dsa(query, key_cache, head_weights, chunk_start_idx=T - Sq)
            indices = topk_large_indices(scores, k, valid_length_tensor)
            out     = kv_cache gathered at indices through page_table_tensor

        ``query`` is ``[B, Hi, Sq, D]``, ``key_cache`` is ``[B, 1, T, D]``, and
        ``head_weights`` is ``[B, 1, Sq, Hi]``. ``kv_cache`` is a block pool
        ``[num_blocks, Hkv, block_size, Dh]`` read through ``page_table_tensor``
        ``[B, max_blocks_per_user]`` INT32, the same layout
        ``paged_scaled_dot_product_attention_decode`` takes. ``cur_pos_tensor``
        ``[B]`` INT32 is each user's current (inclusive) position; rows past it are
        never read. The result is ``[B, Hkv, k, Dh]`` with the dtype and layout of
        ``kv_cache``, so it can be passed as K and V to SDPA when K == V.

        The device kernel is not implemented.

        Args:
            query (ttnn.Tensor): indexer query, ``[B, Hi, Sq, D]``.
            key_cache (ttnn.Tensor): indexer key cache, ``[B, 1, T, D]``.
            head_weights (ttnn.Tensor): folded head scales, ``[B, 1, Sq, Hi]``.
            kv_cache (ttnn.Tensor): paged attention KV pool, ``[num_blocks, Hkv, block_size, Dh]``.
            page_table_tensor (ttnn.Tensor): per-user block ids, ``[B, max_blocks_per_user]`` INT32.
            cur_pos_tensor (ttnn.Tensor): per-user current position, ``[B]`` INT32.
            k (int): number of selected rows.

        Keyword Args:
            valid_length_tensor (Optional[ttnn.Tensor]): 1-element uint32 tensor.
                Score columns at or past this length are not selectable.
            memory_config (Optional[ttnn.MemoryConfig]): output memory config.
                Defaults to interleaved DRAM.
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): score
                compute settings. Defaults to HiFi4 / fp32 dest acc.

        Returns:
            ttnn.Tensor: selected KV rows, ``[B, Hkv, k, Dh]``.
        )doc",
        &ttnn::experimental::deepseek::fused_lightning_select_kv,
        nb::arg("query"),
        nb::arg("key_cache"),
        nb::arg("head_weights"),
        nb::arg("kv_cache"),
        nb::arg("page_table_tensor"),
        nb::arg("cur_pos_tensor"),
        nb::arg("k"),
        nb::kw_only(),
        nb::arg("valid_length_tensor") = std::nullopt,
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::fused_lightning_select_kv::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_fused_lightning_select_kv(::nanobind::module_& mod) {
    fused_lightning_select_kv::detail::bind_fused_lightning_select_kv(mod);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
