// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_lightning_select_kv_nanobind.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

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

            scores[t] = sum_h ReLU(q_h . key_cache[t]) * w_h
            indices   = topk_large_indices(scores, k, valid_length_tensor)
            out       = kv_cache gathered at indices through page_table_tensor

        Decode only: one user (``B == 1``) and one query token (``Sq == 1``).

        ``query`` ``[num_cores, Hi, 1, D]`` and ``head_weights`` ``[num_cores, 1, 1, Hi]``
        are ROW_MAJOR HEIGHT_SHARDED in L1 with one full replica per core (shards
        ``[Hi, D]`` and ``[1, Hi]``), the ``matmul_decode`` rm_hs layout; the op runs on
        their shared shard grid. ``key_cache`` is ``[1, 1, T, D]``. ``kv_cache`` is a
        block pool ``[num_blocks, Hkv, block_size, Dh]`` read through
        ``page_table_tensor`` ``[1, max_blocks_per_user]`` INT32, the same layout
        ``paged_scaled_dot_product_attention_decode`` takes. ``cur_pos_tensor`` ``[1]``
        INT32 is the current (inclusive) position; rows past it are never read. The
        result is ``[1, Hkv, k, Dh]`` with the dtype and layout of ``kv_cache``, so it
        can be passed as K and V to SDPA when K == V.

        The device kernel is not implemented.

        Args:
            query (ttnn.Tensor): indexer query, ``[num_cores, Hi, 1, D]`` ROW_MAJOR
                HEIGHT_SHARDED in L1, one full ``[Hi, D]`` replica per core. The op runs on
                the query's shard grid.
            key_cache (ttnn.Tensor): indexer key cache, ``[1, 1, T, D]``.
            head_weights (ttnn.Tensor): folded head scales, ``[num_cores, 1, 1, Hi]``
                ROW_MAJOR HEIGHT_SHARDED in L1 on the query's shard grid, one full
                ``[1, Hi]`` replica per core.
            kv_cache (ttnn.Tensor): paged attention KV pool, ``[num_blocks, Hkv, block_size, Dh]``.
            page_table_tensor (ttnn.Tensor): block ids, ``[1, max_blocks_per_user]`` INT32.
            cur_pos_tensor (ttnn.Tensor): current position, ``[1]`` INT32.
            k (int): number of selected rows.

        Keyword Args:
            valid_length_tensor (Optional[ttnn.Tensor]): 1-element uint32 tensor.
                Score columns at or past this length are not selectable.
            memory_config (Optional[ttnn.MemoryConfig]): output memory config.
                Defaults to interleaved DRAM.
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): score
                compute settings. Defaults to HiFi4 / fp32 dest acc.

        Returns:
            List[ttnn.Tensor]: ``[kv_rows, scores]``. ``kv_rows`` is ``[1, Hkv, k, Dh]``;
            ``scores`` is the fp32 ROW_MAJOR index score ``[1, 1, 1, max_blocks_per_user * block_size]``,
            valid for the first ``(cur_pos + 1) // 4`` keys.
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
