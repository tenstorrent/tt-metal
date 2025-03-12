// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/unary_backward/unary_backward.hpp"
#include "ttnn/operations/experimental/unary_backward/unary_backward_pybind.hpp"

#include <fmt/format.h>

namespace ttnn::operations::experimental::gelu_backward::detail {
namespace py = pybind11;

void bind_experimental_gelu_backward_operation(py::module& module) {
    auto doc = fmt::format("TODO!");

    using OperationType = decltype(ttnn::experimental::gelu_bw);
    bind_registered_operation(
        module,
        ttnn::experimental::gelu_bw,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& grad_output_tensor,
               const Tensor& input_tensor,
               const string& approximate,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor>& input_grad_tensor) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(grad_output_tensor, input_tensor, approximate, memory_config, input_grad_tensor);
            },
            py::arg("grad_output_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("approximate") = "none",
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad_tensor") = std::nullopt});
}
}  // namespace ttnn::operations::experimental::gelu_backward::detail
