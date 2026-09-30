// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "scatter.hpp"
#include "scatter_enums.hpp"
#include "scatter_force.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_scatter(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Scatters the source tensor's values along a given dimension according to the index tensor.

        Args:
            input (ttnn.Tensor): the input tensor to scatter values onto.
            dim (int): the dimension to scatter along.
            index (ttnn.Tensor): the tensor specifying indices where values from the source tensor must go to.
            src (ttnn.Tensor): the tensor containing the source values to be scattered onto input.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the output tensor. Defaults to `None`.
            reduce (ttnn.ScatterReductionType, optional): reduction operation to apply when multiple values are scattered to the same location (e.g., amax, amin, sum). Currently not supported. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): specifies which cores scatter should run on. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor with scattered values.

        Note:
            * Input tensors must be interleaved and on device.
        )doc";

    ttnn::bind_function<"scatter">(
        mod,
        doc,
        &ttnn::scatter,
        nb::arg("input").noconvert(),
        nb::arg("dim"),
        nb::arg("index").noconvert(),
        nb::arg("src").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("reduce") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());

    // Bound with a plain def rather than ttnn::bind_function: the latter tags the callable for
    // auto_register_ttnn_cpp_operations, which would republish these as ttnn.* operations. They are
    // meant to stay reachable only via this private module. See scatter_force.hpp.
    mod.def(
        "scatter_force_native",
        &scatter_force_native,
        nb::arg("input").noconvert(),
        nb::arg("dim"),
        nb::arg("index").noconvert(),
        nb::arg("src").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("reduce") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the native scatter implementation unconditionally. Not part of the
            ttnn API; use ttnn.scatter, which selects an implementation on its own.
        )doc");

    mod.def(
        "scatter_force_codegen",
        &scatter_force_codegen,
        nb::arg("input").noconvert(),
        nb::arg("dim"),
        nb::arg("index").noconvert(),
        nb::arg("src").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("reduce") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the codegen scatter implementation unconditionally, raising for a
            case outside its support scope rather than falling back to native. Not part of the ttnn
            API; use ttnn.scatter, which selects an implementation on its own.
        )doc");
}

void bind_scatter_add(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Scatters the source tensor's values along a given dimension according to the index tensor, adding source values associated with according repeated indices.

        Args:
            input (ttnn.Tensor): the input tensor to scatter values onto.
            dim (int): the dimension to scatter along.
            index (ttnn.Tensor): the tensor specifying indices where values from the source tensor must go to.
            src (ttnn.Tensor): the tensor containing the source values to be scattered onto input.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the output tensor. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): specifies which cores scatter should run on. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor with scattered values.

        Note:
            * Input tensors must be interleaved and on device.
        )doc";

    ttnn::bind_function<"scatter_add">(
        mod,
        doc,
        &ttnn::scatter_add,
        nb::arg("input").noconvert(),
        nb::arg("dim"),
        nb::arg("index").noconvert(),
        nb::arg("src").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
}

}  // namespace ttnn::operations::data_movement::detail
