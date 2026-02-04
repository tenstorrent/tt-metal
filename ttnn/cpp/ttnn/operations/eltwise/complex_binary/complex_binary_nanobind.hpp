// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// this file doesn't appear to be used anywhere?
#include <string>

#include <fmt/format.h>
#include <nanobind/nanobind.h>

#include "cpp/ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::complex_binary {

namespace detail {

namespace nb = nanobind;

// OpHandler_complex_binary_type1 = get_function_complex_binary
template <typename complex_unary_operation_t>
void bind_complex_binary_type1(
    nb::module_& mod, const complex_unary_operation_t& operation, const std::string& description) {
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
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const complex_unary_operation_t& self,
               const ComplexTensor& input_tensor_a,
               const ComplexTensor& input_tensor_b,
               const ttnn::MemoryConfig& memory_config) -> ComplexTensor {
                return self(input_tensor_a, input_tensor_b, memory_config);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config")});
}

}  // namespace detail

}  // namespace ttnn::operations::complex_binary
