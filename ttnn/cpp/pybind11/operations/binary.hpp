// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../decorators.hpp"
#include "ttnn/operations/binary.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace binary {

namespace detail {

template <typename binary_operation_t>
void bind_binary(py::module& module, const binary_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType>& dtype) -> ttnn::Tensor {
                return self(input_tensor_a, scalar, memory_config, dtype, std::nullopt);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt},
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType>& dtype) -> ttnn::Tensor {
                return self(input_tensor_a, input_tensor_b, memory_config, dtype, std::nullopt);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_binary(
        module,
        ttnn::add,
        R"doc(add(input_tensor_a: ttnn.Tensor, input_tensor_b: Union[ttnn.Tensor, int, float], *, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

        Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`

        .. math:: \mathrm{{ input\_tensor\_a }}_i + \mathrm{{ input\_tensor\_b }}_i

        Supports broadcasting.

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

        Keyword args:
            * :attr:`memory_config` (ttnn.MemoryConfig): memory config for the output tensor
            * :attr:`dtype` (ttnn.DataType): data type for the output tensor

        Example::

            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
            >>> output = ttnn.add(tensor1, tensor2)
            >>> print(output)
            ttnn.Tensor([ 1, 3], dtype=bfloat16))doc");

    detail::bind_binary(
        module,
        ttnn::add_,
        R"doc(add_(input_tensor_a: ttnn.Tensor, input_tensor_b: Union[ttnn.Tensor, int, float], *, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place

.. math::
    \mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i

Supports broadcasting.

Args:
    * :attr:`input_tensor_a`
    * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

Keyword args:
    * :attr:`memory_config` (ttnn.MemoryConfig): memory config for the output tensor
    * :attr:`dtype` (ttnn.DataType): data type for the output tensor

Example::

    >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
    >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
    >>> output = ttnn.add_(tensor1, tensor2)
    >>> print(output)
    ttnn.Tensor([ 1, 3], dtype=bfloat16))doc");
    detail::bind_binary(
        module,
        ttnn::subtract,
        R"doc(subtract(input_tensor_a: ttnn.Tensor, input_tensor_b: Union[ttnn.Tensor, int, float], *, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

        Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a`

        .. math:: \mathrm{{ input\_tensor\_a }}_i - \mathrm{{ input\_tensor\_b }}_i

        Supports broadcasting.

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to subtract from :attr:`input_tensor_a`.
            * :attr:`memory_config` (ttnn.MemoryConfig): memory config for the output tensor
            * :attr:`dtype` (ttnn.DataType): data type for the output tensor

        Example::

                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
                >>> output = ttnn.subtract(tensor1, tensor2)
                >>> print(output)
                ttnn.Tensor([ 1, 1], dtype=bfloat16))doc");
    detail::bind_binary(
        module,
        ttnn::subtract_,
        R"doc(subtract_(input_tensor_a: ttnn.Tensor, input_tensor_b: Union[ttnn.Tensor, int, float], *, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

    Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place

    .. math::
        \mathrm{{input\_tensor\_a}}_i - \mathrm{{input\_tensor\_b}}_i

    Supports broadcasting.

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to subtract from :attr:`input_tensor_a`.
        * :attr:`memory_config` (ttnn.MemoryConfig): memory config for the output tensor
        * :attr:`dtype` (ttnn.DataType): data type for the output tensor

    Example::

            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
            >>> output = ttnn.subtract_(tensor1, tensor2)
            >>> print(output)
            ttnn.Tensor([ 1, 1], dtype=bfloat16))doc");
    detail::bind_binary(
        module,
        ttnn::multiply,
        R"doc(multiply(input_tensor_a: ttnn.Tensor, input_tensor_b: Union[ttnn.Tensor, int, float], *, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

        Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`

        .. math:: \mathrm{{ input\_tensor\_a }}_i \times \mathrm{{ input\_tensor\_b }}_i

        Supports broadcasting.

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to multiply by :attr:`input_tensor_a`.
            * :attr:`memory_config` (ttnn.MemoryConfig): memory config for the output tensor
            * :attr:`dtype` (ttnn.DataType): data type for the output tensor

        Example::

                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
                >>> output = ttnn.multiply(tensor1, tensor2)
                >>> print(output)
                ttnn.Tensor([ 0, 2], dtype=bfloat16))doc");
    detail::bind_binary(
        module,
        ttnn::multiply_,
        R"doc(multiply_(input_tensor_a: ttnn.Tensor, input_tensor_b: Union[ttnn.Tensor, int, float], *, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

    Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place

    .. math::
        \mathrm{{input\_tensor\_a}}_i \times \mathrm{{input\_tensor\_b}}_i

    Supports broadcasting.

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to multiply by :attr:`input_tensor_a`.
        * :attr:`memory_config` (ttnn.MemoryConfig): memory config for the output tensor
        * :attr:`dtype` (ttnn.DataType): data type for the output tensor

    Example::

            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
            >>> output = ttnn.multiply_(tensor1, tensor2)
            >>> print(output)
            ttnn.Tensor([ 0, 2], dtype=bfloat16))doc");
}

}  // namespace binary
}  // namespace operations
}  // namespace ttnn
