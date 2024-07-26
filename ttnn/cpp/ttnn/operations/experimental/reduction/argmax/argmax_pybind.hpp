// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/reduction/argmax/argmax.hpp"

namespace ttnn::operations::experimental::reduction::detail {
namespace py = pybind11;
void bind_argmax_operation(py::module& module) {
    auto doc =
        R"doc(argmax(input_tensor: ttnn.Tensor, *, dim: Optional[int] = None, memory_config: MemoryConfig = std::nullopt) -> ttnn.Tensor

            Returns the indices of the maximum value of elements in the ``input`` tensor
            If no ``dim`` is provided, it will return the indices of maximum value of all elements in given ``input``

            Input tensor must have BFLOAT16 data type and ROW_MAJOR layout.

            Output tensor will have UINT32 data type.

            Equivalent pytorch code:

            .. code-block:: python

                return torch.argmax(input_tensor, dim=dim)

            Args:
                * :attr:`input_tensor`: Input Tensor for argmax.

            Keyword Args:
                * :attr:`dim`: the dimension to reduce. If None, the argmax of the flattened input is returned
                * :attr:`memory_config`: Memory Config of the output tensor
        )doc";

    using OperationType = decltype(ttnn::experimental::argmax);
    bind_registered_operation(
        module,
        ttnn::experimental::argmax,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                int64_t dim,
                bool all,
                const std::optional<ttnn::MemoryConfig>& memory_config) {
                    return self(input_tensor, dim, all, memory_config);
            },
                py::arg("input_tensor").noconvert(),
                py::arg("dim"),
                py::kw_only(),
                py::arg("all") = false,
                py::arg("memory_config") = std::nullopt});
}


void bind_argmin_operation(py::module& module) {
    auto doc =
        R"doc({0}(input_tensor: ttnn.Tensor, dim: int, *, all: bool, memory_config: ttnn.MemoryConfig) -> std::vector<Tensor>

        Performs Argmin function on :attr:`input_tensor`, :attr:`dim`, :attr:`all`.

        Args:
            * :attr:`input_tensor`

        Keyword args:
            * :attr:`dim`
            * :attr:`all` Default is false
            * :attr:`memory_config` [ttnn.MemoryConfig]: memory config for the output tensor

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, dim)
        )doc";

    using OperationType = decltype(ttnn::experimental::argmin);
    bind_registered_operation(
        module,
        ttnn::experimental::argmin,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               int64_t dim,
               bool all,
               const std::optional<MemoryConfig>& memory_config)  {
                return self(input_tensor, dim, all, memory_config);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("all") = false,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::reduction::detail
