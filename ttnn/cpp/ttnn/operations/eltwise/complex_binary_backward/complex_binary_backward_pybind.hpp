// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/complex_binary_backward/complex_binary_backward.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace complex_binary_backward {

namespace detail {

template <typename complex_binary_backward_operation_t>
void bind_complex_binary_backward(py::module& module, const complex_binary_backward_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
R"doc({0}(grad_tensor: ttnn.Tensor, input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, alpha: float, *, memory_config: ttnn.MemoryConfig) -> std::vector<ComplexTensor>

{2}

Args:
    * :attr:`grad_tensor`
    * :attr:`input_tensor_a`
    * :attr:`input_tensor_b`
    * :attr:`alpha`

Keyword args:
    * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor

Example:

    >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
    >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
    >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
    >>> output = {1}(grad_tensor, tensor1, tensor2, alpha)
)doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const complex_binary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor_a,
               const ComplexTensor& input_tensor_b,
               float alpha,
               const ttnn::MemoryConfig& memory_config) -> std::vector<ComplexTensor> {
                return self(grad_tensor, input_tensor_a, input_tensor_b, alpha, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("alpha"),
            py::kw_only(),
            py::arg("memory_config")});
}

}  // namespace detail


void py_module(py::module& module) {
    detail::bind_complex_binary_backward(
        module,
        ttnn::complex_add_bw,
        R"doc(Performs backward operations for addition of :attr:`input_tensor_a` and :attr:`input_tensor_b` complex tensors with given :attr:`grad_tensor`.)doc");

    detail::bind_complex_binary_backward(
        module,
        ttnn::complex_sub_bw,
        R"doc(Performs backward operations for subtraction of :attr:`input_tensor_a` and :attr:`input_tensor_b` complex tensors with given :attr:`grad_tensor`.)doc");

}

}  // namespace binary_backward
}  // namespace operations
}  // namespace ttnn
