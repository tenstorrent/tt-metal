// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/ternary_backward/ternary_backward.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace ternary_backward {

namespace detail {

template <typename ternary_backward_operation_t>
void bind_ternary_backward(py::module& module, const ternary_backward_operation_t& operation, const std::string_view description, const std::string_view supported_dtype = "") {
    auto doc = fmt::format(
        R"doc(

        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor or Number): the input tensor.
            alpha (float, nuber): the alpha value.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Note:
            {3}

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
            >>> tensor3 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, tensor1, tensor2, tensor3, float)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        supported_dtype,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ternary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::Tensor& input_tensor_c,
               float alpha,
               const std::optional<ttnn::MemoryConfig>& memory_config)  {
                auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
                return self(grad_tensor, input_tensor_a, input_tensor_b, input_tensor_c, alpha, output_memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("input_tensor_c"),
            py::arg("alpha"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt}
    );
}

template <typename ternary_backward_operation_t>
void bind_ternary_backward_op(py::module& module, const ternary_backward_operation_t& operation, const std::string_view description, const std::string_view supported_dtype = "") {
    auto doc = fmt::format(
        R"doc(
        {2}


        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor or Number): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.


        Returns:
            List of ttnn.Tensor: the output tensor.


        Note:
            {3}


        Note : bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT


        Example:
            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
            >>> tensor3 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, tensor1, tensor2, tensor3/scalar)

        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        supported_dtype,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ternary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::Tensor& input_tensor_c,
               const std::optional<ttnn::MemoryConfig>& memory_config)  {
                return self(grad_tensor, input_tensor_a, input_tensor_b, input_tensor_c, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("input_tensor_c"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const ternary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const float scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config)  {
                return self(grad_tensor, input_tensor_a, input_tensor_b, scalar, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt}
    );
}

template <typename ternary_backward_operation_t>
void bind_ternary_backward_optional_output(py::module& module, const ternary_backward_operation_t& operation, const std::string_view description, const std::string_view supported_dtype = "") {
    auto doc = fmt::format(
        R"doc(

        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor): the input tensor.

        Keyword args:
            are_required_outputs (List[bool], optional): List of required outputs. Defaults to `[True, True]`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            List of ttnn.Tensor: the output tensor.


        Note:
            {3}

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
            >>> tensor3 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, tensor1, tensor2, tensor3)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        supported_dtype,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ternary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::Tensor& input_tensor_c,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor_a, input_tensor_b, input_tensor_c, memory_config, are_required_outputs, input_a_grad, input_b_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("input_tensor_c"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("input_a_grad") = std::nullopt,
            py::arg("input_b_grad") = std::nullopt,
            py::arg("queue_id") = 0}
    );
}
}  // namespace detail


void py_module(py::module& module) {
    detail::bind_ternary_backward(
        module,
        ttnn::addcmul_bw,
        R"doc(Supported dtypes, layouts, and ranks:

           +----------------------------+---------------------------------+-------------------+
           |     Dtypes                 |         Layouts                 |     Ranks         |
           +----------------------------+---------------------------------+-------------------+
           |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
           +----------------------------+---------------------------------+-------------------+

        )doc",
        R"doc(Performs backward operations for addcmul of :attr:`input_tensor_a` , :attr:`input_tensor_b` and :attr:`input_tensor_c` with given :attr:`grad_tensor`.)doc");

    detail::bind_ternary_backward(
        module,
        ttnn::addcdiv_bw,
        R"doc(Supported dtypes, layouts, and ranks:

           +----------------------------+---------------------------------+-------------------+
           |     Dtypes                 |         Layouts                 |     Ranks         |
           +----------------------------+---------------------------------+-------------------+
           |    BFLOAT16                |       TILE                      |      2, 3, 4      |
           +----------------------------+---------------------------------+-------------------+

        )doc",
        R"doc(Performs backward operations for addcdiv of :attr:`input_tensor_a` , :attr:`input_tensor_b` and :attr:`input_tensor_c` with given :attr:`grad_tensor`.)doc");

    detail::bind_ternary_backward_optional_output(
        module,
        ttnn::where_bw,
        R"doc(Supported dtypes, layouts, and ranks:

           +----------------------------+---------------------------------+-------------------+
           |     Dtypes                 |         Layouts                 |     Ranks         |
           +----------------------------+---------------------------------+-------------------+
           |    BFLOAT16                |       TILE                      |      2, 3, 4      |
           +----------------------------+---------------------------------+-------------------+

        )doc",
        R"doc(Performs backward operations for where of :attr:`input_tensor_a` , :attr:`input_tensor_b` and :attr:`input_tensor_c` with given :attr:`grad_tensor`.)doc");

    detail::bind_ternary_backward_op(
        module,
        ttnn::lerp_bw,
        R"doc(Supported dtypes, layouts, and ranks: For Inputs : :attr:`input_tensor_a` , :attr:`input_tensor_b` and :attr:`input_tensor_c`

           +----------------------------+---------------------------------+-------------------+
           |     Dtypes                 |         Layouts                 |     Ranks         |
           +----------------------------+---------------------------------+-------------------+
           |    BFLOAT16                |       TILE                      |      2, 3, 4      |
           +----------------------------+---------------------------------+-------------------+

        Supported dtypes, layouts, and ranks: For Inputs : :attr:`input_tensor_a` , :attr:`input_tensor_b` and :attr:`scalar`

           +----------------------------+---------------------------------+-------------------+
           |     Dtypes                 |         Layouts                 |     Ranks         |
           +----------------------------+---------------------------------+-------------------+
           |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
           +----------------------------+---------------------------------+-------------------+

        )doc",
        R"doc(Performs backward operations for lerp of :attr:`input_tensor_a` , :attr:`input_tensor_b` and :attr:`input_tensor_c` or :attr:`scalar` with given :attr:`grad_tensor`.)doc");

}

}  // namespace ternary_backward
}  // namespace operations
}  // namespace ttnn
