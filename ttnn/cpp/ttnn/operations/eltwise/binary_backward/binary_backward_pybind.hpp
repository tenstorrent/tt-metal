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
void bind_binary_backward(py::module& module, const binary_backward_operation_t& operation, const std::string& description) {
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
                return self(grad_tensor, input_tensor_a, output_memory_config, input_tensor_b);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},


        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::Tensor& input_tensor_c,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
                using TernaryBackwardOp = ttnn::operations::ternary_backward::ExecuteTernaryBackward<ternary_backward::TernaryBackwardOpType::LERP_BW>;
                return TernaryBackwardOp::execute_on_worker_thread(grad_tensor, input_tensor_a, input_tensor_b, input_tensor_c, output_memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("input_tensor_c"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},


        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const float alpha,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor_a, alpha, input_tensor_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("alpha"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},


        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const string mode,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor_a, mode, input_tensor_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("mode"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},


        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad,
               const uint8_t& queue_id) -> std::vector<optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor_a, input_tensor_b, memory_config, are_required_outputs, input_a_grad, input_b_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("input_a_grad") = std::nullopt,
            py::arg("input_b_grad") = std::nullopt,
            py::arg("queue_id") = 0},

        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad) -> std::vector<optional<ttnn::Tensor>> {
                return self(grad_tensor, input_tensor_a, input_tensor_b, memory_config, are_required_outputs, input_a_grad, input_b_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("input_a_grad") = std::nullopt,
            py::arg("input_b_grad") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const float alpha,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad,
               const uint8_t& queue_id) -> std::vector<optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor_a, input_tensor_b, alpha, memory_config, are_required_outputs, input_a_grad, input_b_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("alpha") = 1.0f,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("input_a_grad") = std::nullopt,
            py::arg("input_b_grad") = std::nullopt,
            py::arg("queue_id") = 0},

        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const float alpha,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad) -> std::vector<optional<ttnn::Tensor>> {
                return self(grad_tensor, input_tensor_a, input_tensor_b, alpha, memory_config, are_required_outputs, input_a_grad, input_b_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("alpha") = 1.0f,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("input_a_grad") = std::nullopt,
            py::arg("input_b_grad") = std::nullopt});
}

}  // namespace detail


void py_module(py::module& module) {
    detail::bind_binary_backward(
        module,
        ttnn::atan2_bw,
        R"doc(Performs backward operations for atan2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::embedding_bw,
        R"doc(Performs backward operations for embedding_bw function and it returns specific indices of the embedding table specified by the :attr:`grad_tensor`.
        The input tensor( :attr:`input_tensor_a`, :attr:`input_tensor_b`) should be unique.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::subalpha_bw,
        R"doc(Performs backward operations for subalpha of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::addalpha_bw,
        R"doc(Performs backward operations for addalpha on :attr:`input_tensor_b` , attr:`input_tensor_a`, attr:`alpha` with given attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::lerp_bw,
        R"doc(Performs backward operations for lerp on :attr:`input_tensor_a` , attr:`input_tensor_b`, attr:`weight` with given attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::xlogy_bw,
        R"doc(Performs backward operations for xlogy of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::hypot_bw,
        R"doc(Performs backward operations for hypot of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::ldexp_bw,
        R"doc(Performs backward operations for ldexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::logaddexp_bw,
        R"doc(Performs backward operations for logaddexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::logaddexp2_bw,
        R"doc(Performs backward operations for logaddexp2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::squared_difference_bw,
        R"doc(Performs backward operations for squared_difference of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::concat_bw,
        R"doc(Performs backward operations for concat on :attr:`input_tensor_a` and :attr:`input_tensor_b` with given attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::rsub_bw,
        R"doc(Performs backward operations for subraction of :attr:`input_tensor_a` from :attr:`input_tensor_b` with given attr:`grad_tensor` (reversed order of subtraction operator).)doc");

    detail::bind_binary_backward(
        module,
        ttnn::bias_gelu_bw,
        R"doc(Performs backward operations for binary gelu of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::binary_ne_bw,
        R"doc(Performs backward operations for not equal comparison on :attr:`input_tensor_a` , attr:`input_tensor_b` with given attr:`grad_tensor`.
        Returns an tensor of zeros like :attr:`input_tensor_a` and :attr:`input_tensor_b` tensor.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::binary_ge_bw,
        R"doc(Performs backward operations for greater than or equal to comparison on :attr:`input_tensor_a` , attr:`input_tensor_b` with given attr:`grad_tensor`.
        Returns an tensor of zeros like :attr:`input_tensor_a` and :attr:`input_tensor_b` tensor.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::min_bw,
        R"doc(Performs backward operations for minimum of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::max_bw,
        R"doc(Performs backward operations for maximum of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward(
        module,
        ttnn::div_bw,
        R"doc(Performs backward operations for divide of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

}

}  // namespace binary_backward
}  // namespace operations
}  // namespace ttnn
