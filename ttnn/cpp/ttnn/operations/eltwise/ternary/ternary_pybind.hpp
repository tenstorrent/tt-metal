// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;


namespace ttnn {
namespace operations {
namespace ternary {

namespace detail {

template <typename ternary_operation_t>
void bind_ternary_composite(py::module& module, const ternary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, input_tensor_c: ttnn.Tensor, *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Args:
                * :attr:`input_tensor_a`
                * :attr:`input_tensor_b`
                * :attr:`input_tensor_c` (ttnn.Tensor or Number):

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

            Example:

                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
                >>> tensor3 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
                >>> output = {1}(tensor1, tensor2, tensor3)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const Tensor& input_tensor_c,
               float value,
               const std::optional<MemoryConfig>& memory_config) {
                    return self(input_tensor_a, input_tensor_b, input_tensor_c, value, memory_config);
                },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("input_tensor_c"),
            py::arg("value") = 1.0f,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

}  // namespace detail

void py_module(py::module& module) {
    // new imported
    detail::bind_ternary_composite(
        module,
        ttnn::addcmul,
        R"doc(compute Addcmul :attr:`input_tensor_a` and :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_ternary_composite(
        module,
        ttnn::addcdiv,
        R"doc(compute Addcdiv :attr:`input_tensor_a` and :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");
}

}  // namespace ternary
}  // namespace operations
}  // namespace ttnn
