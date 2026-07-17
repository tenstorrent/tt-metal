// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "to_layout_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_to_layout(nb::module_& mod) {
    const auto* doc = R"doc(
        Organizes the `ttnn.Tensor` tensor into either `ttnn.ROW_MAJOR_LAYOUT` or `ttnn.TILE_LAYOUT`.

        When requesting `ttnn.ROW_MAJOR_LAYOUT`, the tensor will be returned unpadded in the last two dimensions.
        When requesting `ttnn.TILE_LAYOUT`, the tensor will be padded to the requested tile shape. In the case where
        the layout is the same, the operation simply pads or unpads the last two dimensions depending on the requested
        layout.

        Args:
            tensor (ttnn.Tensor): the input tensor to be organized.
            layout (ttnn.Layout): the desired layout, either `ttnn.ROW_MAJOR_LAYOUT` or `ttnn.TILE_LAYOUT`.
            dtype (ttnn.DataType, optional): the optional output data type.
            memory_config (ttnn.MemoryConfig, optional): the optional output memory configuration.
            sub_core_grids (ttnn.CoreRangeSet, optional): the optional sub core grids. Defaults to `None`.
            pad_value (float, optional): the optional pad value. Defaults to `0.0f`.
            tile (ttnn.Tile, optional): explicit tile metadata for TILE conversions, including
                non-32x32 tiles. Defaults to `None`.

        Returns:
            ttnn.Tensor: the tensor with the requested layout.
        )doc";

    ttnn::bind_function<"to_layout", "ttnn.experimental.quasar.">(
        mod,
        doc,
        &ttnn::operations::experimental::quasar::to_layout,
        nb::arg("tensor"),
        nb::arg("layout"),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("pad_value") = 0.0f,
        nb::kw_only(),
        nb::arg("tile") = nb::none());
}

}  // namespace ttnn::operations::experimental::quasar::detail
