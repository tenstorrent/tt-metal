// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace ternary {

namespace detail {

template <typename ternary_operation_t>
void bind_ternary_composite_float(py::module& module,
                                  const ternary_operation_t& operation,
                                  const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor or Number): the input tensor.


        Keyword Args:
            value (float, optional): Float value. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Supported dtypes and layouts:

            +----------------------------+---------------------------------+-------------------+
            |     Dtypes                 |         Layouts                 |     Ranks         |
            +----------------------------+---------------------------------+-------------------+
            |    BFLOAT16                |          TILE                   |      2, 3, 4      |
            +----------------------------+---------------------------------+-------------------+

            Note : bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

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
        ttnn::pybind_overload_t{[](const ternary_operation_t& self,
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
                                py::kw_only(),
                                py::arg("value") = 1.0f,
                                py::arg("memory_config") = std::nullopt});
}

template <typename ternary_operation_t>
void bind_ternary_where(py::module& module, const ternary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or number): the input tensor.
            input_tensor_c (ttnn.Tensor or Number): the input tensor.


        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.


        Supported dtypes and layouts:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |          TILE                   |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+

        Note : bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

        Example:

            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
            >>> tensor3 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(tensor1, tensor2, tensor3)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{[](const ternary_operation_t& self,
                                   const Tensor& predicate,
                                   const Tensor& true_value,
                                   const Tensor& false_value,
                                   const std::optional<MemoryConfig>& memory_config,
                                   std::optional<Tensor> output_tensor,
                                   uint8_t queue_id) {
                                    return self(
                                        queue_id, predicate, true_value, false_value, memory_config, output_tensor);
                                },
                                py::arg("predicate"),
                                py::arg("true_value"),
                                py::arg("false_value"),
                                py::kw_only(),
                                py::arg("memory_config") = std::nullopt,
                                py::arg("output_tensor") = std::nullopt,
                                py::arg("queue_id") = 0},
        ttnn::pybind_overload_t{[](const ternary_operation_t& self,
                                   const Tensor& predicate,
                                   const float true_value,
                                   const Tensor& false_value,
                                   const std::optional<MemoryConfig>& memory_config,
                                   std::optional<Tensor> output_tensor,
                                   uint8_t queue_id) {
                                    return self(
                                        queue_id, predicate, true_value, false_value, memory_config, output_tensor);
                                },
                                py::arg("predicate"),
                                py::arg("true_value"),
                                py::arg("false_value"),
                                py::kw_only(),
                                py::arg("memory_config") = std::nullopt,
                                py::arg("output_tensor") = std::nullopt,
                                py::arg("queue_id") = 0},
        ttnn::pybind_overload_t{[](const ternary_operation_t& self,
                                   const Tensor& predicate,
                                   const Tensor& true_value,
                                   const float false_value,
                                   const std::optional<MemoryConfig>& memory_config,
                                   std::optional<Tensor> output_tensor,
                                   uint8_t queue_id) {
                                    return self(
                                        queue_id, predicate, true_value, false_value, memory_config, output_tensor);
                                },
                                py::arg("predicate"),
                                py::arg("true_value"),
                                py::arg("false_value"),
                                py::kw_only(),
                                py::arg("memory_config") = std::nullopt,
                                py::arg("output_tensor") = std::nullopt,
                                py::arg("queue_id") = 0},
        ttnn::pybind_overload_t{[](const ternary_operation_t& self,
                                   const Tensor& predicate,
                                   const float true_value,
                                   const float false_value,
                                   const std::optional<MemoryConfig>& memory_config,
                                   std::optional<Tensor> output_tensor,
                                   uint8_t queue_id) {
                                    return self(
                                        queue_id, predicate, true_value, false_value, memory_config, output_tensor);
                                },
                                py::arg("predicate"),
                                py::arg("true_value"),
                                py::arg("false_value"),
                                py::kw_only(),
                                py::arg("memory_config") = std::nullopt,
                                py::arg("output_tensor") = std::nullopt,
                                py::arg("queue_id") = 0});
}

template <typename ternary_operation_t>
void bind_ternary_lerp(py::module& module, const ternary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, input_tensor_c: ttnn.Tensor, *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
            Args:
                * :attr:`input_tensor_a`
                * :attr:`input_tensor_b`
                * :attr:`input_tensor_c` (ttnn.Tensor or Number):

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

            Supported dtypes and layouts:

            +----------------------------+---------------------------------+-------------------+
            |     Dtypes                 |         Layouts                 |     Ranks         |
            +----------------------------+---------------------------------+-------------------+
            |    BFLOAT16, BFLOAT8_B     |          TILE                   |      2, 3, 4      |
            +----------------------------+---------------------------------+-------------------+

            Note : bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

            Example:
                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
                >>> tensor3 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
                >>> output = {1}(tensor1, tensor2, tensor3/scalar)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{[](const ternary_operation_t& self,
                                   const Tensor& input_tensor_a,
                                   const Tensor& input_tensor_b,
                                   const Tensor& input_tensor_c,
                                   const std::optional<MemoryConfig>& memory_config) {
                                    return self(input_tensor_a, input_tensor_b, input_tensor_c, memory_config);
                                },
                                py::arg("input_tensor_a"),
                                py::arg("input_tensor_b"),
                                py::arg("input_tensor_c"),
                                py::kw_only(),
                                py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{[](const ternary_operation_t& self,
                                   const Tensor& input_tensor_a,
                                   const Tensor& input_tensor_b,
                                   float value,
                                   const std::optional<MemoryConfig>& memory_config) {
                                    return self(input_tensor_a, input_tensor_b, value, memory_config);
                                },
                                py::arg("input_tensor_a"),
                                py::arg("input_tensor_b"),
                                py::arg("value") = 1.0f,
                                py::kw_only(),
                                py::arg("memory_config") = std::nullopt});
}

template <typename ternary_operation_t>
void bind_ternary_mac(py::module& module, const ternary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or number): the input tensor.
            input_tensor_c (ttnn.Tensor or Number): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Supported dtypes and layouts:

            +----------------------------+---------------------------------+-------------------+
            |     Dtypes                 |         Layouts                 |     Ranks         |
            +----------------------------+---------------------------------+-------------------+
            |    BFLOAT16                |          TILE                   |      2, 3, 4      |
            +----------------------------+---------------------------------+-------------------+

            Note : bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

        Example:
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
            >>> tensor3 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(tensor1, tensor2/scalar, tensor3/scalar)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{[](const ternary_operation_t& self,
                                   const Tensor& input_tensor_a,
                                   const Tensor& input_tensor_b,
                                   const Tensor& input_tensor_c,
                                   const std::optional<MemoryConfig>& memory_config) {
                                    return self(input_tensor_a, input_tensor_b, input_tensor_c, memory_config);
                                },
                                py::arg("input_tensor_a"),
                                py::arg("input_tensor_b"),
                                py::arg("input_tensor_c"),
                                py::kw_only(),
                                py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{[](const ternary_operation_t& self,
                                   const Tensor& input_tensor_a,
                                   float value1,
                                   float value2,
                                   const std::optional<MemoryConfig>& memory_config) {
                                    return self(input_tensor_a, value1, value2, memory_config);
                                },
                                py::arg("input_tensor_a"),
                                py::arg("value1"),
                                py::arg("value2"),
                                py::kw_only(),
                                py::arg("memory_config") = std::nullopt});
}

}  // namespace detail

void py_module(py::module& module) {
    // new imported
    detail::bind_ternary_composite_float(
        module,
        ttnn::addcmul,
        R"doc(compute Addcmul :attr:`input_tensor_a` and :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    detail::bind_ternary_composite_float(
        module,
        ttnn::addcdiv,
        R"doc(compute Addcdiv :attr:`input_tensor_a` and :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    detail::bind_ternary_where(
        module,
        ttnn::where,
        R"doc(compute Addcdiv :attr:`input_tensor_a` and :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    detail::bind_ternary_lerp(
        module,
        ttnn::lerp,
        R"doc(compute Lerp :attr:`input_tensor_a` and :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_ternary_mac(
        module,
        ttnn::mac,
        R"doc(compute Mac :attr:`input_tensor_a` and :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");
}

}  // namespace ternary
}  // namespace operations
}  // namespace ttnn
