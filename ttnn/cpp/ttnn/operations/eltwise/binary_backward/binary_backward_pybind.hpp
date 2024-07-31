// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "ttnn/operations/eltwise/ternary_backward/ternary_backward.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace binary_backward {

namespace detail {

template <typename binary_backward_operation_t>
void bind_binary_backward_ops(py::module& module, const binary_backward_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc({0}(grad_tensor: ttnn.Tensor, input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig) -> std::vector<Tensor>

{2}

Args:
    * :attr:`grad_tensor`
    * :attr:`input_tensor_a`
    * :attr:`input_tensor_b`

Keyword args:
    * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor

Example:

    >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
    >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
    >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
    >>> output = {1}(grad_tensor, tensor1, tensor2)
)doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
                return self(grad_tensor, input_tensor_a, input_tensor_b, output_memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_int_default(py::module& module, const binary_backward_operation_t& operation, const std::string& parameter_name, const std::string& parameter_doc, int parameter_value, const std::string& description) {
    auto doc = fmt::format(
        R"doc({0}(grad_tensor: ttnn.Tensor, input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, {2}: int, *, memory_config: ttnn.MemoryConfig) -> std::vector<Tensor>

        {5}

        Args:
            * :attr:`grad_tensor`
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`
            * :attr:`{2}` (int):`{3}`, Default value = {4}

        Keyword args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
            >>> output = {1}(grad_tensor, tensor1, tensor2, int)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        parameter_value,
        description);


    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               int parameter,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor_a, input_tensor_b, parameter, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg(parameter_name.c_str()) = parameter_value,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt}
    );
}

template <typename binary_backward_operation_t>
void bind_binary_backward_opt_float_default(py::module& module, const binary_backward_operation_t& operation, const std::string& parameter_name, const std::string& parameter_doc, float parameter_value, const std::string& description) {
    auto doc = fmt::format(
        R"doc({0}(grad_tensor: ttnn.Tensor, input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, {2}: float, *, memory_config: ttnn.MemoryConfig) -> std::vector<Tensor>

        {5}

        Args:
            * :attr:`grad_tensor`
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`
            * :attr:`{2}` (float):`{3}`,Default value = {4}

        Keyword args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
            >>> output = {1}(grad_tensor, tensor1, tensor2, float)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        parameter_value,
        description);


    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               float parameter,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad,
               const uint8_t& queue_id) -> std::vector<optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor_a, input_tensor_b, parameter, memory_config, are_required_outputs, input_a_grad, input_b_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg(parameter_name.c_str()) = parameter_value,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("input_a_grad") = std::nullopt,
            py::arg("input_b_grad") = std::nullopt,
            py::arg("queue_id") = 0}
    );
}

template <typename binary_backward_operation_t>
void bind_binary_backward_float_default(py::module& module, const binary_backward_operation_t& operation, const std::string& parameter_name, const std::string& parameter_doc, float parameter_value, const std::string& description) {
    auto doc = fmt::format(
        R"doc({0}(grad_tensor: ttnn.Tensor, input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, {2}: float, *, memory_config: ttnn.MemoryConfig) -> std::vector<Tensor>

        {5}

        Args:
            * :attr:`grad_tensor`
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`
            * :attr:`{2}` (float):`{3}`,Default value = {4}

        Keyword args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
            >>> output = {1}(grad_tensor, tensor1, tensor2, float)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        parameter_value,
        description);


    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               float parameter,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor_a, input_tensor_b, parameter, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg(parameter_name.c_str()) = parameter_value,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt}
    );
}

template <typename binary_backward_operation_t>
void bind_binary_bw_mul(py::module& module, const binary_backward_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor_a: Union[ttnn.Tensor, ComplexTensor] , input_tensor_b: Union[ComplexTensor, ttnn.Tensor, int, float], *, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None, activations: Optional[List[str]] = None) -> ttnn.Tensor or ComplexTensor

        {2}
        Supports broadcasting.

        Args:
            * :attr:`input_tensor_a` (ComplexTensor or ttnn.Tensor)
            * :attr:`input_tensor_b` (ComplexTensor or ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

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
            [](const binary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const Tensor& input_tensor_a,
               const float scalar,
               const std::optional<MemoryConfig>& memory_config){
                return self(grad_tensor, input_tensor_a, scalar, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& other_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad,
               const uint8_t& queue_id) -> std::vector<optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor, other_tensor, memory_config, are_required_outputs, input_grad, other_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg("other_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("input_grad") = std::nullopt,
            py::arg("other_grad") = std::nullopt,
            py::arg("queue_id") = 0},

        // complex tensor
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor_a,
               const ComplexTensor& input_tensor_b,
               const MemoryConfig& memory_config) {
                return self(grad_tensor, input_tensor_a, input_tensor_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_float(py::module& module, const binary_backward_operation_t& operation, const std::string& parameter_name, const std::string& parameter_doc, const std::string& description) {
    auto doc = fmt::format(
        R"doc({0}(grad_tensor: ttnn.Tensor, input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, {2}: float, *, memory_config: ttnn.MemoryConfig) -> std::vector<Tensor>

        {4}

        Args:
            * :attr:`grad_tensor`
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`
            * :attr:{2} {3}

        Keyword args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
            >>> output = {1}(grad_tensor, tensor1, tensor2, float)
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
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               float parameter,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor_a, input_tensor_b, parameter, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg(parameter_name.c_str()),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

}  // namespace detail


void py_module(py::module& module) {
    detail::bind_binary_bw_mul(
        module,
        ttnn::mul_bw,
        R"doc(Performs backward operations for multiply on :attr:`input_tensor`, :attr:`alpha` or attr:`input_tensor_a`, attr:`input_tensor_b`, with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::atan2_bw,
        R"doc(Performs backward operations for atan2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::embedding_bw,
        R"doc(Performs backward operations for embedding_bw function and it returns specific indices of the embedding table specified by the :attr:`grad_tensor`.
        The input tensor( :attr:`input_tensor_a`, :attr:`input_tensor_b`) should be unique.)doc");

    detail::bind_binary_backward_float_default(
        module,
        ttnn::subalpha_bw,
        "alpha", "Alpha value", 1.0f,
        R"doc(Performs backward operations for subalpha of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_opt_float_default(
        module,
        ttnn::addalpha_bw,
        "alpha", "Alpha value", 1.0f,
        R"doc(Performs backward operations for addalpha on :attr:`input_tensor_b` , attr:`input_tensor_a`, attr:`alpha` with given attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::xlogy_bw,
        R"doc(Performs backward operations for xlogy of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::hypot_bw,
        R"doc(Performs backward operations for hypot of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::ldexp_bw,
        R"doc(Performs backward operations for ldexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::logaddexp_bw,
        R"doc(Performs backward operations for logaddexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::logaddexp2_bw,
        R"doc(Performs backward operations for logaddexp2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::squared_difference_bw,
        R"doc(Performs backward operations for squared_difference of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_int_default(
        module,
        ttnn::concat_bw,
        "dim", "Dimension to concatenate", 0,
        R"doc(Performs backward operations for concat on :attr:`input_tensor_a` and :attr:`input_tensor_b` with given attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::rsub_bw,
        R"doc(Performs backward operations for subraction of :attr:`input_tensor_a` from :attr:`input_tensor_b` with given attr:`grad_tensor` (reversed order of subtraction operator).)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::min_bw,
        R"doc(Performs backward operations for minimum of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::max_bw,
        R"doc(Performs backward operations for maximum of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

}

}  // namespace binary_backward
}  // namespace operations
}  // namespace ttnn
