// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "fill_pad.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement {
namespace {

void bind_fill_pad_op(nb::module_& mod) {
    const auto* doc = R"doc(
        Fills the implicit padding of a tiled input tensor with the specified value.
        Specifically, any nD tensor will have the implicit padding of the last 2 dims that exists from [height:tile_height, width:tile_width] filled with the specified value.

        Args:
            input_tensor (ttnn.Tensor): Any input tensor with desired device and data types for output tensor.
            fill_value (float): Value to fill the tensor padding with.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"fill_implicit_tile_padding">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::fill_implicit_tile_padding,
            nb::arg("input_tensor"),
            nb::arg("fill_value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace

void bind_fill_pad(nb::module_& mod) { bind_fill_pad_op(mod); }

}  // namespace ttnn::operations::data_movement
