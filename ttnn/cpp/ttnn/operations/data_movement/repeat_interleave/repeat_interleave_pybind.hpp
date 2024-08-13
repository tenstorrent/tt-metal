// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "repeat_interleave.hpp"



namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_repeat_interleave(py::module& module) {
    auto doc =
    R"doc(repeat_interleave(input_tensor: ttnn.Tensor, repeats : int, dim: int = 0) -> ttnn.Tensor

        Repeats elements of a :attr:`tensor` in the given :attr:`dim`.

        Args:
            * :attr:`input_tensor`: the input_tensor to apply the repeate interleave operation.
            * :attr:`repeats`: The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
            * :attr:`dim`: the dimension to expand with the repetitions.

        Example:

        torch_input_tensor =
            torch_result = torch.repeat_interleave(torch_input_tensor, repeats, dim=dim)

            input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

            output = ttnn.repeat_interleave(input_tensor, repeats, dim=dim)
            >>> a = ttnn.from_torch(torch.rand(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> b = ttnn.repeat_interleave(a, 2, dim=0)
            >>> print(a.shape, b.shape)
            ttnn.Shape([1, 1, 32, 32]) ttnn.Shape([2, 1, 32, 32])
        )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::repeat_interleave,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("repeats"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}


} // namespace ttnn::operations::data_movement::detail
