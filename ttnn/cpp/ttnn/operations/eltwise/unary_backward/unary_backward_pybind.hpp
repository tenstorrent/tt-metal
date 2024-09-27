// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace unary_backward {

namespace detail {

template <typename unary_backward_operation_t>
void bind_unary_backward_two_float(
    py::module& module, const unary_backward_operation_t& operation, std::string_view description, std::string_view supported_dtype) {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor (ttnn.Tensor): the input tensor.
            threshold (float): the input threshold value.
            value (float): the input value.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        {3}

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input, float, float)
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
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float min,
               float max,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, min, max, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg("min"),
            py::arg("max"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_op(
    py::module& module, const unary_backward_operation_t& operation, const std::string& description, const std::string& supported_dtype) {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        {3}

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input)
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
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_rsqrt(
    py::module& module, const unary_backward_operation_t& operation, std::string_view description, std::string_view supported_dtype) {
    auto doc = fmt::format(
        R"doc(

        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        {3}

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input)
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
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor, memory_config, input_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_op_reciprocal(
    py::module& module, const unary_backward_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ComplexTensor or ttnn.Tensor): the input tensor.
            input_tensor_a (ComplexTensor or ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor,
               const MemoryConfig& memory_config) {
                return self(grad_tensor, input_tensor, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config")});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_op_overload_abs(
    py::module& module, const unary_backward_operation_t& operation, std::string_view description) {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor (ComplexTensor or ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const ComplexTensor& input_tensor,
               const MemoryConfig& memory_config) {
                return self(grad_tensor, input_tensor, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config")});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_float(py::module& module, const unary_backward_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor (ttnn.Tensor): the input tensor.
            float_value (Number): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input, float)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float scalar,
               const std::optional<MemoryConfig>& memory_config)  {
                return self(grad_tensor, input_tensor, scalar, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt}

    );
}

template <typename unary_backward_operation_t>
void bind_unary_backward_two_float_with_default(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    float parameter_a_value,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    float parameter_b_value,
    std::string_view description) {
    auto doc = fmt::format(
        R"doc(
        {8}

        Args:
            grad_tensor (ComplexTensor or ttnn.Tensor): the input tensor.
            input_tensor (ComplexTensor or ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float, optional): {3} , Default to {4}
            {5} (float, optional): {6} , Default to {7}
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input, {2} = {3}, {5} = {6})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               float parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_float_with_default(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    float parameter_a_value,
    const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {5}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.


        Keyword args:
            {2} (float, optional): {3} , Defaults to {4}
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input, {2} = {3})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_optional_float_params_with_default(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    std::optional<float> parameter_a_value,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    std::optional<float> parameter_b_value,
    const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {8}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float, optional): {3} , Default value = {4}
            {5} (float, optional): {6} , Default value = {7}
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input, {2} = {3}, {5} = {6})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               std::optional<float> parameter_a,
               std::optional<float> parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_float_string_default(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    string parameter_b_value,
    std::string_view description) {
    auto doc = fmt::format(
        R"doc(
        {7}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (float): the input scalar.


        Keyword args:
            round_mode (round_mode, optional): Round mode for the operation. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> output = {1}(grad_tensor, input, {2}, {4} = {6})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               string parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()),
            py::kw_only(),
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_string_default(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    string parameter_a_value,
    std::string_view description) {
    auto doc = fmt::format(
        R"doc(
        {5}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (string, optional): {3} , Defaults to {4}
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input, {2} = {4})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               string parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_unary_optional_float(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    std::string_view description) {
    auto doc = fmt::format(
        R"doc(
        {4}

        Args:
            grad_tensor (ttnn.Tensor): the input grad tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): Command queue id. Defaults to `0`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> output = {1}(grad_tensor, tensor, `{2}`)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    queue_id, grad_tensor, input_tensor, parameter, memory_config, input_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg(parameter_name.c_str()),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_shape(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    std::string_view description) {
    auto doc = fmt::format(
        R"doc(
        {4}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (string): {3} of tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input, {2})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::LegacyShape& parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_optional(
    py::module& module, const unary_backward_operation_t& operation, std::string_view description, std::string_view supported_dtype = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ComplexTensor or ttnn.Tensor): the input tensor.
            input_tensor_a (ComplexTensor or ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        {3}

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor, memory_config, input_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_prod_bw(py::module& module, const unary_backward_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Performs backward operations for prod on input along `all_dimensions` or a particular `dim`.

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.

        Keyword args:
            all_dimensions (bool, optional): perform prod backward along all dimensions ,ignores dim param . Defaults to `True`.
            dim (int, optional): Dimension to perform prod backward. Defaults to `0`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input, all_dimensions, dim)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               bool all_dimensions,
               int64_t dim,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, all_dimensions, dim, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("all_dimensions") = true,
            py::arg("dim") = 0,
            py::arg("memory_config") = std::nullopt});
}


template <typename unary_backward_operation_t>
void bind_unary_backward(
    py::module& module, const unary_backward_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,

        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
                return self(grad_tensor, input_tensor, output_memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_gelu(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    string parameter_a_value,
    std::string_view description) {
    auto doc = fmt::format(
        R"doc(

        {5}

        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            approximate (ttnn.Tensor, optional): Approximation type.  Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input, {2} = {4})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        description);
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               string parameter_a,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor, parameter_a, memory_config, input_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_unary_backward_optional_float_params_with_default(
        module,
        ttnn::clamp_bw,
        "min",
        "Minimum value",
        std::nullopt,
        "max",
        "Maximum value",
        std::nullopt,
        R"doc(Performs backward operations for clamp value on :attr:`input_tensor`, :attr:`min`, :attr:`max` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_two_float_with_default(
        module,
        ttnn::hardtanh_bw,
        "min",
        "Minimum value",
        -1.0,
        "max",
        "Maximum value",
        1.0,
        R"doc(Performs backward operations for hardtanh activation function on :attr:`input_tensor`, :attr:`min`, :attr:`max` with given :attr:`grad_tensor`.)doc");


    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::hardshrink_bw,
        "lambd",
        "Lambda value for the hardshrink formula ",
        0.5,
        R"doc(Performs backward operations for hardshrink on :attr:`input_tensor`, :attr:`lambd`, with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::softshrink_bw,
        "lambd",
        "Lambda value for the softshrink formula ",
        0.5,
        R"doc(Performs backward operations for softshrink on :attr:`input_tensor`, :attr:`lambd`, with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::leaky_relu_bw,
        "negative_slope",
        "negative_slope value for the hardshrink formula ",
        0.01,
        R"doc(Performs backward operations for leaky_relu on :attr:`input_tensor`, :attr:`negative_slope`, with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::elu_bw,
        "alpha",
        "alpha value for the elu formula ",
        1.0,
        R"doc(Performs backward operations for elu on :attr:`input_tensor`, :attr:`alpha`, with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::celu_bw,
        "alpha",
        "alpha value for the celu formula ",
        1.0,
        R"doc(Performs backward operations for celu on :attr:`input_tensor`, :attr:`alpha`, with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::logiteps_bw,
        "eps",
        "eps value for the logiteps formula ",
        0.0,
        R"doc(Performs backward operations for logiteps on :attr:`input_tensor`, :attr:`eps`, with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_two_float(
        module,
        ttnn::threshold_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for threshold on :attr:`input_tensor`, :attr:`threshold`, :attr:`value` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_two_float_with_default(
        module,
        ttnn::softplus_bw,
        "beta",
        "Beta value for the Softplus formula ",
        1.0,
        "threshold",
        "Threshold value",
        20.0,
        R"doc(Performs backward operations for softplus on :attr:`input_tensor`, :attr:`beta`, :attr:`threshold` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_float_string_default(
        module,
        ttnn::rdiv_bw,
        "scalar",
        "divisor",
        "round_mode",
        "Mode of Rounding",
        "None",
        R"doc(Performs backward operations for Unary rdiv on :attr:`input_tensor`, :attr:`scalar` with given :attr:`grad_tensor` using given :attr:`round_mode`.
        :attr:`round_mode` can be 'None', 'trunc', or 'floor'.)doc");

    detail::bind_unary_backward_shape(
        module,
        ttnn::repeat_bw,
        "shape",
        "Shape",
        R"doc(Performs backward operations for repeat on :attr:`input_tensor_a` or :attr:`input_tensor`, with given :attr:`grad_tensor` using given :attr:`shape`.)doc");

    detail::bind_unary_backward_gelu(
        module,
        ttnn::gelu_bw,
        "approximate",
        "Approximation type",
        "none",
        R"doc(Performs backward operations for gelu on :attr:`input_tensor_a` or :attr:`input_tensor`, with given :attr:`grad_tensor` using given :attr:`approximate` mode.
        :attr:`approximate` mode can be 'none', 'tanh'.)doc");

    detail::bind_unary_backward_unary_optional_float(
        module,
        ttnn::pow_bw,
        "exponent",
        "Exponent value",
        R"doc(Performs backward operations for power on :attr:`input_tensor` , :attr:`exponent` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_optional(
        module,
        ttnn::exp_bw,
        R"doc(Performs backward operations for exponential function on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_optional(
        module,
        ttnn::tanh_bw,
        R"doc(Performs backward operations for Hyperbolic Tangent (Tanh) function on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_optional(
        module,
        ttnn::sqrt_bw,
        R"doc(Performs backward operations for square-root on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::multigammaln_bw,
        R"doc(Performs backward operations for multigammaln on :attr:`input_tensor` with given :attr:`grad_tensor` and value of P is taken as 4.
        mvlgamma is refered as multigammaln.
        Input value must be greater than 2.5f)doc");

    detail::bind_unary_backward_prod_bw(module, ttnn::prod_bw);

    detail::bind_unary_backward(
        module,
        ttnn::lgamma_bw,
        R"doc(Performs backward operations for lgamma on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_optional(
        module,
        ttnn::fill_bw,
        R"doc(Performs backward operations for fill on :attr:`input_tensor` with given :attr:`grad_tensor`.
        Returns an tensor like :attr:`grad_tensor` with sum of tensor values.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::hardsigmoid_bw,
        R"doc(Performs backward operations for hardsigmoid on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::cos_bw,
        R"doc(Performs backward operations for cos on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::acosh_bw,
        R"doc(Performs backward operations for inverse cosine (acos) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::acos_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for inverse hyperbolic cosine (acosh) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::atan_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for atan on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::rad2deg_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for radian to degree conversion (rad2deg) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::frac_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE, ROW_MAJOR           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for frac on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::trunc_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE, ROW_MAJOR           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for truncation on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::log_sigmoid_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for log sigmoid on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::fill_zero_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE, ROW_MAJOR           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations of fill zero on :attr:`input_tensor` with given :attr:`grad_tensor`. Returns an tensor of zeros like :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::i0_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for i0 on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::tan_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for tan on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::sigmoid_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for sigmoid on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_rsqrt(
        module,
        ttnn::rsqrt_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for rsqrt on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_optional(
        module,
        ttnn::neg_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for neg on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::relu_bw,
        R"doc(Performs backward operations for relu on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::logit_bw,
        R"doc(Performs backward operations for logit on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::floor_bw,
        R"doc(Performs backward operations for floor on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_float(
        module,
        ttnn::rpow_bw,
        R"doc(Performs backward operations for rpow on :attr:`input_tensor`, :attr:`exponent` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::round_bw,
        R"doc(Performs backward operations for round on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::log_bw,
        R"doc(Performs backward operations for logarithm on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::relu6_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for relu6 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op_overload_abs(
        module,
        ttnn::abs_bw,
        R"doc(Performs backward operations for abs on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_optional(
        module,
        ttnn::silu_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for silu on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::selu_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for selu on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::square_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for square on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::hardswish_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for  hardswish on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::tanhshrink_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for  tanhshrink on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::atanh_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for  atanh on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::asin_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for  asin on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::asinh_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for  asinh on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::sin_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for sin on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::sinh_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for sinh on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::log10_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for log10 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::log1p_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for log1p on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::erfc_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for erfc on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::ceil_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE, ROW_MAJOR           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for ceil on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::softsign_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for softsign on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::cosh_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for cosh on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::log2_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for log2 on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::sign_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE, ROW_MAJOR           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for sign on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_float(
        module,
        ttnn::div_no_nan_bw,
        R"doc(Performs backward operations for div_no_nan on :attr:`input_tensor`, :attr:`scalar` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::exp2_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for exp2 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::expm1_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for exp2 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op_reciprocal(
        module,
        ttnn::reciprocal_bw,
        R"doc(Performs backward operations for reciprocal on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::digamma_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for digamma on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::erfinv_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for erfinv on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::erf_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for erf on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::deg2rad_bw,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc",
        R"doc(Performs backward operations for deg2rad on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_float(
        module,
        ttnn::polygamma_bw,
        R"doc(Performs backward operations for polygamma on :attr:`input_tensor` or attr:`input_tensor_a`, attr:`scalar` with given :attr:`grad_tensor`.)doc");
}

}  // namespace unary_backward
}  // namespace operations
}  // namespace ttnn
