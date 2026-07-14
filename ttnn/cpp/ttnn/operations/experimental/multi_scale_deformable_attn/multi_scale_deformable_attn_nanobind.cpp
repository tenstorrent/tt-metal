// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_scale_deformable_attn_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/multi_scale_deformable_attn/multi_scale_deformable_attn.hpp"

namespace ttnn::operations::experimental::multi_scale_deformable_attn::detail {

void bind_multi_scale_deformable_attn(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused multi-scale deformable attention (num_levels == 1 fast path).

        Args:
            * :attr:`value`: (N, h_in, w_in, D) ROW_MAJOR bfloat16, N = B * num_heads
            * :attr:`grid`: (N, Q*P, 1, 2) ROW_MAJOR bfloat16, normalized to [-1, 1]
            * :attr:`attn`: (N, Q, P) ROW_MAJOR bfloat16
            * :attr:`memory_config`: output memory config
            * :attr:`align_corners`: bilinear pixel-coord mapping
                - False (default, matches mmcv): pixel = (g + 1) * size / 2 - 0.5
                - True:                           pixel = (g + 1) * (size - 1) / 2

        Returns:
            (N, Q, D) ROW_MAJOR bfloat16.
        )doc";
    ttnn::bind_function<"multi_scale_deformable_attn", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::multi_scale_deformable_attn,
        nb::arg("value"),
        nb::arg("grid"),
        nb::arg("attn"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("align_corners") = false);
}

}  // namespace ttnn::operations::experimental::multi_scale_deformable_attn::detail
