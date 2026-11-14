// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compress_pool_decode_nanobind.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/compress_pool_decode/compress_pool_decode.hpp"

namespace ttnn::operations::experimental::deepseek::detail {

void bind_compress_pool_decode(::nanobind::module_& mod) {
    namespace nb = nanobind;
    ttnn::bind_function<"compress_pool_decode", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Fused DeepSeek-V4.1-Flash single-row compressor window update + pool + RMSNorm + optional RoPE.

        Replaces the multi-op single-row compressor chain in ``DeepSeekV41Attention._latent`` and
        ``_apply_rope`` across ``Dh // 32`` worker cores (16 cores for ``Dh = 512``):

            - When window caches (``gate``, ``win_kv``, ``win_gate``, ``win_slot``) are provided:
              writes ``(kv, gate)`` into row ``win_slot`` of ``(win_kv, win_gate)`` (in 32x32 TILE DRAM,
              preserving other rows in the tile). If ``do_pool`` is true, reads ``(k0, g0)`` at
              ``prev_slot`` (default ``1 - win_slot``) and computes:
                  w0 = sigmoid(g0 - gate)
                  x  = kv + w0 * (k0 - kv)
            - When window caches are omitted (compress_ratio == 1, ``do_pool=True``): ``x = kv``.
            - Computes ``latent = rms_norm(x, norm_weight, epsilon)``.
            - If ``cos`` and ``sin`` are provided, also computes ``roped_row = rope(latent, cos, sin)``
              on the trailing ``rope_dim`` channels.

        Args:
            kv (ttnn.Tensor): ``[1, 1, 1, Dh]`` bfloat16 ROW_MAJOR (DRAM interleaved or L1 WIDTH_SHARDED).
            norm_weight (ttnn.Tensor): ``[1, 1, 1, Dh]`` bfloat16 (32x32 TILE or ROW_MAJOR DRAM).
            epsilon (float): RMSNorm epsilon (default ``1e-6``).
            gate (Optional[ttnn.Tensor]): ``[1, 1, 1, Dh]`` bfloat16 ROW_MAJOR (DRAM or L1 WIDTH_SHARDED).
            win_kv (Optional[ttnn.Tensor]): ``[1, 1, rows, Dh]`` bfloat16 32x32 TILE DRAM.
            win_gate (Optional[ttnn.Tensor]): ``[1, 1, rows, Dh]`` bfloat16 32x32 TILE DRAM.
            win_slot (Optional[ttnn.Tensor]): ``[1]`` int32 ROW_MAJOR slot index to write.
            prev_slot (Optional[ttnn.Tensor]): ``[1]`` int32 ROW_MAJOR slot index to pair with
                (defaults to ``1 - win_slot`` when omitted).
            cos (Optional[ttnn.Tensor]): ``[1, 1, 1, rope_dim]`` bfloat16 32x32 TILE DRAM.
            sin (Optional[ttnn.Tensor]): ``[1, 1, 1, rope_dim]`` bfloat16 32x32 TILE DRAM.
            do_pool (bool): whether to pool, RMSNorm, and emit outputs (default ``True``).

        With ``cos`` / ``sin`` the op can also write the closed group's entry in place
        (``Dk = index_head_dim``, a multiple of 32 at least ``rope_dim``):

            index_wk (Optional[ttnn.Tensor]): ``[Dh, Dk]`` (K, N) bfloat16 32x32 TILE DRAM.
            index_norm_weight (Optional[ttnn.Tensor]): ``[1, 1, 1, Dk]`` bfloat16 (TILE or ROW_MAJOR DRAM).
            idx_key_cache (Optional[ttnn.Tensor]): ``[n_blocks, 1, block_rows, Dk]`` bfloat16 TILE DRAM; gets
                ``rope(rms_norm(latent @ index_wk, index_norm_weight, epsilon))`` at logical row ``entry``.
            comp_kv (Optional[ttnn.Tensor]): ``[n_blocks, 1, block_rows, Dh]`` bfloat16 ROW_MAJOR DRAM; gets
                ``roped_row`` at logical row ``entry``.
            page_table (Optional[ttnn.Tensor]): ``[rows, n_blocks]`` int32 ROW_MAJOR DRAM mapping both pools'
                logical blocks to physical ones.
            entry (Optional[ttnn.Tensor]): ``[1]`` int32 logical row of ``idx_key_cache`` / ``comp_kv``.
            page_row (Optional[ttnn.Tensor]): ``[1]`` int32 row of ``page_table`` (default 0).
            dense_kv (Optional[ttnn.Tensor]): ``[1, 1, rows, Dh]`` bfloat16 TILE DRAM; gets ``roped_row``.
            dense_entry (Optional[ttnn.Tensor]): ``[1]`` int32 row of ``dense_kv``; ``-1`` skips the write.
            key_row (Optional[ttnn.Tensor]): ``[1, 1, 1, Dk]`` bfloat16 ROW_MAJOR DRAM; also gets the key.

        Returns:
            List[ttnn.Tensor]:
                - ``[]`` if ``do_pool`` is ``False``.
                - ``[latent]`` (``[1, 1, 1, Dh]`` bfloat16 32x32 TILE DRAM) if ``cos``/``sin`` omitted.
                - ``[latent, roped_row]`` where ``roped_row`` is ``[1, 1, 1, Dh]`` bfloat16 ROW_MAJOR DRAM
                  if ``cos`` and ``sin`` are provided.
        )doc",
        &ttnn::experimental::deepseek::compress_pool_decode,
        nb::arg("kv"),
        nb::arg("norm_weight"),
        nb::arg("epsilon") = 1e-6f,
        nb::arg("gate") = std::nullopt,
        nb::arg("win_kv") = std::nullopt,
        nb::arg("win_gate") = std::nullopt,
        nb::arg("win_slot") = std::nullopt,
        nb::arg("prev_slot") = std::nullopt,
        nb::arg("cos") = std::nullopt,
        nb::arg("sin") = std::nullopt,
        nb::arg("do_pool") = true,
        nb::arg("index_wk") = std::nullopt,
        nb::arg("index_norm_weight") = std::nullopt,
        nb::arg("idx_key_cache") = std::nullopt,
        nb::arg("comp_kv") = std::nullopt,
        nb::arg("page_table") = std::nullopt,
        nb::arg("entry") = std::nullopt,
        nb::arg("page_row") = std::nullopt,
        nb::arg("dense_kv") = std::nullopt,
        nb::arg("dense_entry") = std::nullopt,
        nb::arg("key_row") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
