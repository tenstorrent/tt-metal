// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_pool_gather_nanobind.hpp"

#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "paged_pool_gather.hpp"

namespace ttnn::operations::experimental::paged_pool_gather::detail {

void bind_paged_pool_gather(nb::module_& mod) {
    ttnn::bind_function<"paged_pool_gather", "ttnn.experimental.">(
        mod,
        R"doc(
        Gather pools of ``pool_size`` consecutive token rows out of a paged TILE cache into a compact
        paged TILE cache (one contiguous page range per user, identity page table), e.g. the top-k pools
        of a DSA indexer selection plus the current incomplete pool, which the dense paged flash-MLA
        decode then attends with a per-user ``cur_pos``.

        Output row ``pool_size*i + j`` of user ``b`` (pages ``[b*out_blocks, (b+1)*out_blocks)``) is cache row
        ``pool_size*pool_ids[b, i] + j`` of user ``b`` (logical row, translated through ``page_table[b]``).
        A pool id of 0xFFFFFFFF or past the user's page range yields zero rows; the rows past
        ``pool_size*N`` in the last output block are zero. Pure data movement: a pool is read and
        written as ``pool_size``-row runs inside each tile face, so a token row costs 1 KB of DRAM traffic
        on each side regardless of the cache size.

        Args:
            cache (ttnn.Tensor):      [max_blocks, 1, 32, W] bf16 TILE, DRAM interleaved (the paged cache).
            page_table (ttnn.Tensor): [B_table, blocks_per_seq] int32 ROW_MAJOR; row b maps user b's logical blocks.
            pool_ids (ttnn.Tensor):   [B, 1, 1, N] uint32 ROW_MAJOR, B <= B_table; pool ids per user.

        Keyword args:
            pool_size (int): rows per pool (1, 2, 4, 8 or 16; a pool never crosses a 16-row tile face). Default 4.
            memory_config (ttnn.MemoryConfig, optional): output memory config (DRAM interleaved by default).

        Returns:
            ttnn.Tensor: [B * out_blocks, 1, 32, W] bf16 TILE with out_blocks = ceil(N * pool_size / 32).
        )doc",
        &ttnn::experimental::paged_pool_gather,
        nb::arg("cache").noconvert(),
        nb::arg("page_table").noconvert(),
        nb::arg("pool_ids").noconvert(),
        nb::kw_only(),
        nb::arg("pool_size") = 4,
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::paged_pool_gather::detail
