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
            * :attr:`value`: (N, h_in, w_in, D) ROW_MAJOR bfloat16, N = B * num_heads. With
              num_heads > 1 it is (B, h_in, w_in, num_heads*D) instead: the reader takes head
              n % num_heads out of the stick by byte offset, so no head-major copy is needed.
            * :attr:`grid`: (N, Q, 1, P*2) ROW_MAJOR bfloat16, (x, y) interleaved per point, normalized to [-1, 1]. (N, Q*P, 1, 2) is also accepted but costs P NoC reads per query instead of one. Rank 3
              (B, Q, num_heads*stride*2) packs every head and level, read with num_points and
              point_offset like attn.
            * :attr:`attn`: (N, Q, P) ROW_MAJOR bfloat16. With num_heads > 1 it may instead be
              (B, Q, num_heads*stride): a head's run starts at h*stride and this call reads
              num_points points from point_offset into it.
            * :attr:`memory_config`: output memory config
            * :attr:`align_corners`: bilinear pixel-coord mapping
                - False (default, matches mmcv): pixel = (g + 1) * size / 2 - 0.5
                - True:                           pixel = (g + 1) * (size - 1) / 2
            * :attr:`num_heads`: heads packed into value's last dimension (default 1)
            * :attr:`num_points`: points sampled per query; 0 (default) takes attn's last dim
            * :attr:`point_offset`: points to skip into each head's run, in packed attn and grid
              (default 0)

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
        nb::arg("align_corners") = false,
        nb::arg("num_heads") = 1,
        nb::arg("num_points") = 0,
        nb::arg("point_offset") = 0);
}

}  // namespace ttnn::operations::experimental::multi_scale_deformable_attn::detail
