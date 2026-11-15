// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flash_mqa_decode_nanobind.hpp"

#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/flash_mqa_decode/flash_mqa_decode.hpp"

namespace ttnn::operations::experimental::deepseek::detail {

void bind_flash_mqa_decode(::nanobind::module_& mod) {
    namespace nb = nanobind;
    ttnn::bind_function<"flash_mqa_decode", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Single-token MQA decode attention for DeepSeek V4.1: one shared K == V head, per-head
        attention sinks, over two KV segments, the sliding-window ring ``kv`` and the CSA top-k
        ``selected_kv``. The kernels are the DeepSeek FlashMLA / blaze ``HcaFlashMQADecode`` SDPA
        kernels: 8 heads per core, the chunks of both segments spread over ``num_s_blocks`` cores per
        head group and merged with a tree reduction, with the sink merged as one more peer.

        Args:
            q (ttnn.Tensor): ``[1, 1, heads, head_dim]`` bfloat16 row-major; heads a multiple of 8.
            kv (ttnn.Tensor): ``[1, 1, rows, head_dim]`` TILE bfloat16 / bfloat8_b; rows a multiple of
                ``k_chunk_size``.
            lengths (ttnn.Tensor): row-major int32 whose first row starts with
                ``[ring_valid, selected_valid]`` and, optionally, ``ring_first_row`` and then
                ``selected_first_row``. Attends ``kv[ring_first_row:ring_first_row + ring_valid]`` and
                ``selected_valid`` rows of ``selected_kv`` from row ``selected_row_offset + selected_first_row``.
                ``ring_valid`` must be at least 1, ``ring_first_row`` and ``selected_first_row`` multiples
                of 32. Read on device, so it may change under a trace.

        Keyword Args:
            selected_kv (Optional[ttnn.Tensor]): ``[1, 1, rows, head_dim]`` TILE with kv's dtype.
            selected_row_offset (int): row of ``selected_kv`` holding selected row 0, a multiple of 32.
            ring_rows (int): most ring rows attended (one session's ring in a shared ``kv``), a multiple
                of ``k_chunk_size``; 0 for all of kv's rows.
            selected_rows (int): most selected rows attended (one session's rows in a shared
                ``selected_kv``), a multiple of ``k_chunk_size``; 0 for all rows past ``selected_row_offset``.
            attention_sink (Optional[ttnn.Tensor]): bfloat16 TILE ``[heads, 32]``, ``sink / scale`` in
                column 0.
            scale (Optional[float]): softmax scale, default ``head_dim ** -0.5``.
            k_chunk_size (int): rows per chunk, a multiple of 64.
            num_s_blocks (int): cores per head group.
            memory_config (Optional[ttnn.MemoryConfig]): output memory config, default q's.
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): fp32 dest accumulation
                and full-sync dst are not supported.
            cos (Optional[ttnn.Tensor]): bfloat16 32x32 TILE ``[1, 1, >=1, rope_dim]`` RoPE cosines
                (``rope_dim`` 32 or 64). When set with ``sin``, applies interleaved RoPE in place to
                the trailing ``rope_dim`` columns (row 0 of ``cos``/``sin``) before writing output.
            sin (Optional[ttnn.Tensor]): bfloat16 32x32 TILE ``[1, 1, >=1, rope_dim]`` RoPE sines
                (pass ``-sin`` for inverse RoPE).

            new_kv (Optional[ttnn.Tensor]): block mode only: TILE ``[1, 1, 32, head_dim]`` with kv's dtype, rows
                ``0 .. num_rows - 1`` = the block's new kv rows (not yet in the ring); the other rows must be finite.
            num_rows (int): query rows per head, 1 (one token) or a DSpark verify block of 2, 4 or 8 rows at
                positions ``p .. p + num_rows - 1``, causal inside the block. Block mode: ``q`` is
                ``[1, 1, heads * num_rows, head_dim]`` row-major with row ``h * num_rows + j``; ``lengths`` is int32
                ``[1, 1, 1, 4 + num_rows]`` = ``[p, ring_first_row, selected_first_row, 0, n_0, ..., n_{R-1}]``;
                row ``j`` attends the ring slots (``position % ring_capacity``) of positions
                ``max(0, p + j - ring_capacity + 1) .. p - 1``, ``new_kv`` rows ``0 .. j`` and ``n_j`` selected rows;
                ``cos`` / ``sin`` are required, TILE ``[1, 1, num_rows, rope_dim]`` (tile ``(num_rows, 32)`` or
                32x32), row ``j`` for block row ``j``. Returns TILE ``(num_rows, 32)``
                ``[1, 1, num_rows, heads * head_dim]``.
            dense_selected (bool): block mode: row ``j``'s selected rows are the first ``n_j`` rows of
                ``selected_kv`` from ``selected_row_offset + selected_first_row`` (a shared ``dense_kv``) instead of
                its own block starting ``j * selected_capacity`` rows later (per-row selections;
                ``selected_capacity`` = ``selected_rows``, or the rows past the offset / ``num_rows`` when 0).

            row_groups (int): block mode: the rows run on ``row_groups`` groups of ``num_s_blocks`` x head-group
                cores, group ``rg`` attending rows ``rg, rg + row_groups, ...`` (a chunk shared by rows is read once
                per group); 0 picks the most that divides ``num_rows`` and fits the grid's height.

        Returns:
            ttnn.Tensor: ``[1, 1, heads, head_dim]`` bfloat16 row-major (block mode: see ``num_rows``).
        )doc",
        &ttnn::experimental::deepseek::flash_mqa_decode,
        nb::arg("q"),
        nb::arg("kv"),
        nb::arg("lengths"),
        nb::kw_only(),
        nb::arg("selected_kv") = std::nullopt,
        nb::arg("selected_row_offset") = 0,
        nb::arg("ring_rows") = 0,
        nb::arg("selected_rows") = 0,
        nb::arg("attention_sink") = std::nullopt,
        nb::arg("scale") = std::nullopt,
        nb::arg("k_chunk_size") = 64,
        nb::arg("num_s_blocks") = 8,
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt,
        nb::arg("cos") = std::nullopt,
        nb::arg("sin") = std::nullopt,
        nb::arg("new_kv") = std::nullopt,
        nb::arg("num_rows") = 1,
        nb::arg("dense_selected") = false,
        nb::arg("row_groups") = 0);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
