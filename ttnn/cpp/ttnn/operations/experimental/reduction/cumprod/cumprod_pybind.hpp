// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/reduction/cumprod/cumprod.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::reduction::cumprod::detail {
namespace py = pybind11;
void bind_cumprod_operation(py::module& module) {
    // TODO(jbbieniek): finish this
    auto doc =
        R"doc(

        cumprod(input_tensor: ttnn.Tensor, dim: int) -> ttnn.Tensor

        Returns a tensor witth cumulative product calculated along a given axis (`dim`).

        Args:
            ###

        Keyword Args:
            ###

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            ###
        )doc";

    using OperationType = decltype(ttnn::experimental::cumprod);
    bind_registered_operation(
        module,
        ttnn::experimental::cumprod,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self, const ttnn::Tensor& input_tensor, int64_t dim) {
                return self(input_tensor, dim);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim")});
}

}  // namespace ttnn::operations::experimental::reduction::cumprod::detail
