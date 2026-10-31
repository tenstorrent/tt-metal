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
                ``[ring_valid, selected_valid]`` and, optionally, ``ring_first_row``. Attends
                ``kv[ring_first_row:ring_first_row + ring_valid]`` and
                ``selected_kv[selected_row_offset:selected_row_offset + selected_valid]``.
                ``ring_valid`` must be at least 1 and ``ring_first_row`` a multiple of 32. Read on device,
                so it may change under a trace.

        Keyword Args:
            selected_kv (Optional[ttnn.Tensor]): ``[1, 1, rows, head_dim]`` TILE with kv's dtype.
            selected_row_offset (int): row of ``selected_kv`` holding selected row 0, a multiple of 32.
            ring_rows (int): most ring rows attended (one session's ring in a shared ``kv``), a multiple
                of ``k_chunk_size``; 0 for all of kv's rows.
            attention_sink (Optional[ttnn.Tensor]): bfloat16 TILE ``[heads, 32]``, ``sink / scale`` in
                column 0.
            scale (Optional[float]): softmax scale, default ``head_dim ** -0.5``.
            k_chunk_size (int): rows per chunk, a multiple of 64.
            num_s_blocks (int): cores per head group.
            memory_config (Optional[ttnn.MemoryConfig]): output memory config, default q's.
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): fp32 dest accumulation
                and full-sync dst are not supported.

        Returns:
            ttnn.Tensor: ``[1, 1, heads, head_dim]`` bfloat16 row-major.
        )doc",
        &ttnn::experimental::deepseek::flash_mqa_decode,
        nb::arg("q"),
        nb::arg("kv"),
        nb::arg("lengths"),
        nb::kw_only(),
        nb::arg("selected_kv") = std::nullopt,
        nb::arg("selected_row_offset") = 0,
        nb::arg("ring_rows") = 0,
        nb::arg("attention_sink") = std::nullopt,
        nb::arg("scale") = std::nullopt,
        nb::arg("k_chunk_size") = 64,
        nb::arg("num_s_blocks") = 8,
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
