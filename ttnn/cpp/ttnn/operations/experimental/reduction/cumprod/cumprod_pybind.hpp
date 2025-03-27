// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
    auto doc =
        R"doc(

        cumprod(input_tensor: ttnn.Tensor, dim: int) -> ttnn.Tensor

        Returns a tensor witth cumulative product calculated along a given axis (`dim`).

        Args:
            input_tensor (ttnn.Tensor): the input tensor to calculate cumulative product of.
            dim (int): direction of product cumulation

        Returns:
            ttnn.Tensor: the output tensor (for now, it is a copy of input_tensor, because only scaffold is implemented).

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2, 3), dtype=torch.bfloat16), device=device)
            >>> # Note that the call below will output the same tensor it was fed for the time being,
            >>> # until the actual implementation is provided.
            >>> output = ttnn.experimental.cumprod(tensor, 1)
            >>> assert tensor.shape == output.shape
        )doc";

    using OperationType = decltype(ttnn::experimental::cumprod);
    bind_registered_operation(
        module,
        ttnn::experimental::cumprod,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self, const ttnn::Tensor& input_tensor, const int32_t dim) {
                return self(input_tensor, dim);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim")});
}

}  // namespace ttnn::operations::experimental::reduction::cumprod::detail
