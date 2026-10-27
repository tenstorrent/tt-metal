// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "build_candidate_pool_nanobind.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/build_candidate_pool/build_candidate_pool.hpp"

namespace ttnn::operations::experimental::deepseek::detail {

void bind_build_candidate_pool(::nanobind::module_& mod) {
    namespace nb = nanobind;
    ttnn::bind_function<"build_candidate_pool", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Compacts the candidate blocks chosen by a ``fused_lightning_select_kv`` candidate source
        into a pool that later layers select from with the same op::

            pool_keys[j] = key_cache at logical key candidate_blocks[j // cbs] * cbs + j % cbs
            pool_rows[j] = kv cache row of the same logical key

        Both go through ``page_table_tensor``, which the kv cache shares. Only the first
        ``pool_count`` entries (rounded up to whole candidate blocks) are written; the rest keep
        their previous contents. A consumer then calls ``fused_lightning_select_kv(query, pool_keys,
        ..., kv_cache, identity_page_table, ..., num_keys_tensor=pool_count, row_table=pool_rows)``.

        Args:
            candidate_blocks (ttnn.Tensor): ``[1, candidate_topk_blocks]`` INT32 kept blocks, ascending.
            pool_count (ttnn.Tensor): ``[1]`` INT32 number of valid keys in them.
            key_cache (ttnn.Tensor): bfloat16 TILE DRAM ``[num_blocks, 1, block_size, D]``.
            page_table_tensor (ttnn.Tensor): ``[1, max_blocks_per_user]`` INT32.
            candidate_block_size (int): keys per candidate block, 2, 4, 8 or 16.

        Keyword Args:
            pool_keys_output (Optional[ttnn.Tensor]): preallocated pool keys.
            pool_rows_output (Optional[ttnn.Tensor]): preallocated pool rows.
            page_table_row_tensor (Optional[ttnn.Tensor]): ``[1]`` INT32 row of a
                ``[num_sessions, max_blocks_per_user]`` page table to read, on device.

        Returns:
            List[ttnn.Tensor]: ``[pool_keys, pool_rows]``: ``[candidate_topk_blocks * cbs / block_size,
            1, block_size, D]`` TILE with the key cache's dtype and ``[1, candidate_topk_blocks * cbs]``
            INT32 row-major, both DRAM interleaved.
        )doc",
        &ttnn::experimental::deepseek::build_candidate_pool,
        nb::arg("candidate_blocks"),
        nb::arg("pool_count"),
        nb::arg("key_cache"),
        nb::arg("page_table_tensor"),
        nb::arg("candidate_block_size"),
        nb::kw_only(),
        nb::arg("pool_keys_output") = std::nullopt,
        nb::arg("pool_rows_output") = std::nullopt,
        nb::arg("page_table_row_tensor") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
