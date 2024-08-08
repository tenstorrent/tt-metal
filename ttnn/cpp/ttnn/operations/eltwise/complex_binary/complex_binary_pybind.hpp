// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace complex_binary {

namespace detail {

//OpHandler_complex_binary_type1 = get_function_complex_binary
template <typename complex_unary_operation_t>
void bind_complex_binary_type1(py::module& module, const complex_unary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
R"doc({0}(input_tensor_a: ComplexTensor, input_tensor_b: ComplexTensor, *, memory_config: ttnn.MemoryConfig) -> ComplexTensor

{2}

Args:
    * :attr:`input_tensor_a` (ComplexTensor)
    * :attr:`input_tensor_b` (ComplexTensor)

Keyword args:
    * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor

Example:

    >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
    >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
    >>> output = {1}(tensor1, tensor2)
)doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const complex_unary_operation_t& self,
               const ComplexTensor& input_tensor_a,
               const ComplexTensor& input_tensor_b,
               const ttnn::MemoryConfig& memory_config) -> ComplexTensor {
                return self(input_tensor_a, input_tensor_b, memory_config);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config")});
}

}  // namespace detail

}  // namespace complex_binary
}  // namespace operations
}  // namespace ttnn
