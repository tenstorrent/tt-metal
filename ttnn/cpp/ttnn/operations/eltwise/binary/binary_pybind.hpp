// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::binary {

namespace detail {

template <typename binary_operation_t>
void bind_binary_operation(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor_a: ttnn.Tensor, input_tensor_b: Union[ttnn.Tensor, int, float], *, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None, activations: Optional[List[str]] = None) -> ttnn.Tensor

        {2}

        Supports broadcasting.

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

        Keyword args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor
            * :attr:`dtype` (Optional[ttnn.DataType]): data type for the output tensor
            * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
            * :attr:`activations` (Optional[List[str]]): list of activation functions to apply to the output tensor
            * :attr:`queue_id` (Optional[uint8]): command queue id

        Example:

            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
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
        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<FusedActivations>& activations,
               const uint8_t& queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor_a, scalar, dtype, memory_config, output_tensor, activations);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = std::nullopt,
            py::arg("queue_id") = 0},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<FusedActivations>& activations,
               const uint8_t& queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor_a, input_tensor_b, dtype, memory_config, output_tensor, activations);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_binary_operation(
        module,
        ttnn::add,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`

        .. math:: \mathrm{{ input\_tensor\_a }}_i + \mathrm{{ input\_tensor\_b }}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::add_,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::subtract,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{ input\_tensor\_a }}_i - \mathrm{{ input\_tensor\_b }}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::subtract_,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i - \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::multiply,
        R"doc(Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{ input\_tensor\_a }}_i \times \mathrm{{ input\_tensor\_b }}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::multiply_,
        R"doc(Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i \times \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::eq,
        R"doc(Compares if :attr:`input_tensor_a` is equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i == \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::ne,
        R"doc(Compares if :attr:`input_tensor_a` is not equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i != \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::lt,
        R"doc(Compares if :attr:`input_tensor_a` is less than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i < \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::le,
        R"doc(MCompares if :attr:`input_tensor_a` is less than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i <= \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::gt,
        R"doc(Compares if :attr:`input_tensor_a` is greater than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i > \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::ge,
        R"doc(Compares if :attr:`input_tensor_a` is greater than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i >= \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::logical_and,
        R"doc(Compute logical AND of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i && \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::logical_or,
        R"doc(Compute logical OR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::ldexp,
        R"doc(Compute ldexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::logaddexp,
        R"doc(Compute logaddexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::logaddexp2,
        R"doc(Compute logaddexp2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::squared_difference,
        R"doc(Compute squared difference of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::bias_gelu,
        R"doc(Compute bias_gelu of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::divide,
        R"doc(Divides :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");
}

}
