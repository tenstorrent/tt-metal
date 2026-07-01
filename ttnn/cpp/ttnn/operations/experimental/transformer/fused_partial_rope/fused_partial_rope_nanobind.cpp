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

        Applies interleaved RoPE to the trailing ``rope_dim`` channels of a height-sharded
        ``[1, 1, rows, D]`` input and passes the leading ``D - rope_dim`` "nope" channels
        through untouched, all in a single device op::

            out[..., :D-Rd] = x[..., :D-Rd]
            out[..., D-Rd:] = x_rope * cos + (x_rope @ trans_mat) * sin

        One tile-row (32 rows) is processed per core, so the op uses ``ceil(rows / 32)`` cores.

        Args:
            input (ttnn.Tensor): height-sharded ``[1, 1, rows, D]`` device tensor, TILE layout.
            cos (ttnn.Tensor): height-sharded ``[1, 1, rows, rope_dim]`` cos table, same shard grid.
            sin (ttnn.Tensor): height-sharded ``[1, 1, rows, rope_dim]`` sin table, same shard grid.
            trans_mat (ttnn.Tensor): single ``[32, 32]`` ``rotate_half`` tile (replicated).
            rope_dim (int): trailing channel count that gets RoPE (tile-aligned).

        Keyword Args:
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
        nb::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::transformer
