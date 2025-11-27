// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad_nanobind.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "fill_pad.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::data_movement {
namespace {

void bind_fill_pad_op(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(

            Fills the implicit padding of a tiled input tensor with the specified value.
            Specifically, any nD tensor will have the implicit padding of the last 2 dims that exists from [height:tile_height, width:tile_width] filled with the specified value.

            +----------+-----------------------------------------+-----------------------+------------------------+----------+
            | Argument     | Description                         | Data type             | Valid range            | Required |
            +==========+=========================================+=======================+========================+==========+
            | input_tensor | A tiled input tensor                | tt_lib.tensor.Tensor  |                        | Yes      |
            +----------+-----------------------------------------+-----------------------+------------------------+----------+
            | fill_value   | value to fill into padding          | float                 | [-inf , inf]           | Yes      |
            +----------+-----------------------------------------+-----------------------+------------------------+----------+

            Args:
                input_tensor (ttnn.tensor): Any input tensor with desired device and data types for output tensor.
                value greater than 0 fill_value (float): Value to fill the tensor padding with.

            Keyword args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor.

        )doc",
        ttnn::fill_implicit_tile_padding.base_name());

    using OperationType = decltype(ttnn::fill_implicit_tile_padding);
    ttnn::bind_registered_operation(
        mod,
        ttnn::fill_implicit_tile_padding,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const Tensor& input_tensor,
               const float fill_value,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, fill_value, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("fill_value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace

void bind_fill_pad(nb::module_& mod) { bind_fill_pad_op(mod); }

}  // namespace ttnn::operations::data_movement
