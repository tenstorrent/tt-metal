// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::reduction::detail {

template <typename reduction_operation_t>
void bind_reduction_operation(py::module& module, const reduction_operation_t& operation) {
    namespace py = pybind11;
    auto doc = fmt::format(
        R"doc(

            Args:
                input_a (ttnn.Tensor): the input tensor.
                dim (number): dimension value .

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor.

            Example:

                >>> input_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
                >>> output = ttnn.{0}(input_a, dim, memory_config)
        )doc",
        operation.base_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("dim") = std::nullopt,
            py::arg("keepdim") = true,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("scalar") = 1.0f});
}

}  // namespace ttnn::operations::reduction::detail
