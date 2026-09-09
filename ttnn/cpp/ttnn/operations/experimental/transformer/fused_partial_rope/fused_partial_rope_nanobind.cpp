// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_partial_rope_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "fused_partial_rope.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_fused_partial_rope(nb::module_& mod) {
    ttnn::bind_function<"fused_partial_rope", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused partial rotary position embedding (deepseek_v4_flash ``_apply_rope``).

        Applies interleaved RoPE independently to each ``head_dim``-wide block of a
        height- or width-sharded ``[1, 1, rows, D]`` input (``D`` must be a multiple of
        ``head_dim``; omit or pass 0 to use ``D`` as a single block). Within each block
        the trailing ``rope_dim`` channels are rotated and the leading
        ``head_dim - rope_dim`` "nope" channels pass through, all in a single device op::

            for each block of head_dim channels:
                out[..., :Hd-Rd] = x[..., :Hd-Rd]
                out[..., Hd-Rd:] = x_rope * cos + (x_rope @ trans_mat) * sin

        TILE input is processed as 32x32 tiles. ROW_MAJOR input is processed as 1x32
        faces (one row of 32 elements per tile) and requires a single broadcast
        ``cos``/``sin`` row. The output layout matches the input.

        Args:
            input (ttnn.Tensor): height- or width-sharded ``[1, 1, rows, D]`` device
                tensor, TILE or ROW_MAJOR layout.
            cos (ttnn.Tensor): DRAM-interleaved ``[1, 1, rows, rope_dim]`` (or one
                broadcast row) cos table, TILE layout. Shared across every head block
                of a row.
            sin (ttnn.Tensor): DRAM-interleaved ``[1, 1, rows, rope_dim]`` (or one
                broadcast row) sin table, TILE layout.
            trans_mat (ttnn.Tensor): single ``[32, 32]`` ``rotate_half`` tile (replicated).
            rope_dim (int): trailing channel count of each head block that gets RoPE
                (tile-aligned).

        Keyword Args:
            head_dim (int): block width along the last dim. Defaults to ``D``.
            memory_config (Optional[ttnn.MemoryConfig]): output memory config. Defaults to the
                input's memory config.
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): compute settings.
                Defaults to ``HiFi4``.

        Returns:
            ttnn.Tensor: a new tensor with the same spec as ``input``, partially rotary-embedded.
        )doc",
        &ttnn::experimental::fused_partial_rope,
        nb::arg("input"),
        nb::arg("cos"),
        nb::arg("sin"),
        nb::arg("trans_mat"),
        nb::arg("rope_dim"),
        nb::kw_only(),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt,
        nb::arg("head_dim") = 0);
}

}  // namespace ttnn::operations::experimental::transformer
