// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/binary.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace binary {

namespace detail {

template <typename binary_operation_t>
void bind_binary(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor_a: ttnn.Tensor, input_tensor_b: Union[ttnn.Tensor, int, float], *, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None, activations: Optional[List[str]] = None) -> ttnn.Tensor

        {2}

        Supports broadcasting.

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

        Keyword args:
            * :attr:`memory_config` (ttnn.MemoryConfig): memory config for the output tensor
            * :attr:`dtype` (ttnn.DataType): data type for the output tensor
            * :attr:`activations` (List[str]): list of activation functions to apply to the output tensor

        Example::

            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
            >>> output = {1}(tensor1, tensor2)
        )doc",
        operation.name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType>& dtype,
               const std::optional<std::vector<std::string>>& activations) -> ttnn::Tensor {
                return self(input_tensor_a, scalar, memory_config, dtype, activations);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("activations") = std::nullopt},
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType>& dtype,
               const std::optional<std::vector<std::string>>& activations) -> ttnn::Tensor {
                return self(input_tensor_a, input_tensor_b, memory_config, dtype, activations);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("activations") = std::nullopt});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_binary(
        module,
        ttnn::add,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`

        .. math:: \mathrm{{ input\_tensor\_a }}_i + \mathrm{{ input\_tensor\_b }}_i)doc");

    detail::bind_binary(
        module,
        ttnn::add_,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary(
        module,
        ttnn::subtract,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{ input\_tensor\_a }}_i - \mathrm{{ input\_tensor\_b }}_i)doc");

    detail::bind_binary(
        module,
        ttnn::subtract_,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i - \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary(
        module,
        ttnn::multiply,
        R"doc(Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{ input\_tensor\_a }}_i \times \mathrm{{ input\_tensor\_b }}_i)doc");

    detail::bind_binary(
        module,
        ttnn::multiply_,
        R"doc(Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i \times \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary(
        module,
        ttnn::eq,
        R"doc(Compares if :attr:`input_tensor_a` is equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i == \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary(
        module,
        ttnn::ne,
        R"doc(Compares if :attr:`input_tensor_a` is not equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i != \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary(
        module,
        ttnn::lt,
        R"doc(Compares if :attr:`input_tensor_a` is less than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i < \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary(
        module,
        ttnn::le,
        R"doc(MCompares if :attr:`input_tensor_a` is less than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i <= \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary(
        module,
        ttnn::gt,
        R"doc(Compares if :attr:`input_tensor_a` is greater than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i > \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary(
        module,
        ttnn::ge,
        R"doc(Compares if :attr:`input_tensor_a` is greater than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i >= \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary(
        module,
        ttnn::logical_and,
        R"doc(Compute logical AND of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i && \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary(
        module,
        ttnn::logical_or,
        R"doc(Compute logical OR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");
}

}  // namespace binary
}  // namespace operations
}  // namespace ttnn
