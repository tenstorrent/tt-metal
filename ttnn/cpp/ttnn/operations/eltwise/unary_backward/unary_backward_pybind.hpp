// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace unary_backward {

namespace detail {

template <typename unary_backward_operation_t>
void bind_unary_backward(py::module& module, const unary_backward_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
R"doc({0}(grad_tensor: ttnn.Tensor, input_tensor: ttnn.Tensor *, memory_config: ttnn.MemoryConfig) -> std::vector<Tensor>

{2}

Args:
    * :attr:`grad_tensor`
    * :attr:`input_tensor`

Keyword args:
    * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor

Example:

    >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
    >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
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
            [operation](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());

                using BinaryBackwardOp = ttnn::operations::binary_backward::ExecuteBinaryBackward<binary_backward::BinaryBackwardOpType::MUL_BW>;
                if(operation.base_name()=="assign_bw"){
                    using BinaryBackwardOp = ttnn::operations::binary_backward::ExecuteBinaryBackward<binary_backward::BinaryBackwardOpType::ASSIGN_BW>;
                    return BinaryBackwardOp::execute_on_worker_thread(grad_tensor, input_tensor_a, output_memory_config, input_tensor_b);
                }else if(operation.base_name()=="add_bw"){
                    using BinaryBackwardOp = ttnn::operations::binary_backward::ExecuteBinaryBackward<binary_backward::BinaryBackwardOpType::ADD_BW>;
                    return BinaryBackwardOp::execute_on_worker_thread(grad_tensor, input_tensor_a, output_memory_config, input_tensor_b);
                }else if(operation.base_name()=="eq_bw"){
                    using BinaryBackwardOp = ttnn::operations::binary_backward::ExecuteBinaryBackward<binary_backward::BinaryBackwardOpType::EQ_BW>;
                    return BinaryBackwardOp::execute_on_worker_thread(grad_tensor, input_tensor_a, output_memory_config, input_tensor_b);
                }else if(operation.base_name()=="sub_bw"){
                    using BinaryBackwardOp = ttnn::operations::binary_backward::ExecuteBinaryBackward<binary_backward::BinaryBackwardOpType::SUB_BW>;
                    return BinaryBackwardOp::execute_on_worker_thread(grad_tensor, input_tensor_a, output_memory_config, input_tensor_b);
                }
                return BinaryBackwardOp::execute_on_worker_thread(grad_tensor, input_tensor_a, output_memory_config, input_tensor_b);

            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [operation](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad,
               const uint8_t& queue_id) -> std::vector<optional<ttnn::Tensor>> {
                using BinaryBackwardOp = ttnn::operations::binary_backward::ExecuteBinaryBackward<binary_backward::BinaryBackwardOpType::MUL_BW>;
                if(operation.base_name()=="add_bw"){
                    using BinaryBackwardOp = ttnn::operations::binary_backward::ExecuteBinaryBackward<binary_backward::BinaryBackwardOpType::ADD_BW>;
                    return BinaryBackwardOp::execute_on_main_thread(queue_id, grad_tensor, input_tensor_a, input_tensor_b, memory_config, are_required_outputs, input_a_grad, input_b_grad);
                }else if(operation.base_name()=="eq_bw"){
                    using BinaryBackwardOp = ttnn::operations::binary_backward::ExecuteBinaryBackward<binary_backward::BinaryBackwardOpType::EQ_BW>;
                    return BinaryBackwardOp::execute_on_main_thread(queue_id, grad_tensor, input_tensor_a, input_tensor_b, memory_config, are_required_outputs, input_a_grad, input_b_grad);
                }
                return BinaryBackwardOp::execute_on_main_thread(queue_id, grad_tensor, input_tensor_a, input_tensor_b, memory_config, are_required_outputs, input_a_grad, input_b_grad);
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
            py::arg("memory_config") = std::nullopt},


        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const float alpha,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor, alpha, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg("alpha"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},


        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const float a,
               const float b,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor, a, b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg("a"),
            py::arg("b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});

}

}  // namespace detail


void py_module(py::module& module) {
    detail::bind_unary_backward(
        module,
        ttnn::mul_bw,
        R"doc(Performs backward operations for multiply on :attr:`input_tensor`, :attr:`alpha` or attr:`input_tensor_a`, attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::clamp_min_bw,
        R"doc(Performs backward operations for clamp min value on :attr:`input_tensor`, :attr:`alpha` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::clamp_bw,
        R"doc(Performs backward operations for clamp value on :attr:`input_tensor`, :attr:`min`, :attr:`max` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::assign_bw,
        R"doc(Performs backward operations for assign on :attr:`input_tensor` or attr:`input_tensor_a`, attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::multigammaln_bw,
        R"doc(Performs backward operations for multigammaln on :attr:`input_tensor` with given :attr:`grad_tensor` and value of P is taken as 4.
        mvlgamma is refered as multigammaln.
        Input value must be greater than 2.5f)doc");

    detail::bind_unary_backward(
        module,
        ttnn::add_bw,
        R"doc(Performs backward operations for addition on :attr:`input_tensor`, :attr:`alpha` or attr:`input_tensor_a`, attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::eq_bw,
        R"doc(Performs backward operations for equal to comparison on :attr:`input_tensor`, :attr:`alpha` or attr:`input_tensor_a`, attr:`input_tensor_b` with given :attr:`grad_tensor`.
        Returns an tensor of zeros like input tensors.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::lgamma_bw,
        R"doc(Performs backward operations for lgamma on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
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

    detail::bind_unary_backward(
        module,
        ttnn::acos_bw,
        R"doc(Performs backward operations for inverse hyperbolic cosine (acosh) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::atan_bw,
        R"doc(Performs backward operations for atan on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::rad2deg_bw,
        R"doc(Performs backward operations for radian to degree conversion (rad2deg) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::sub_bw,
        R"doc(Performs backward operations for subtraction on :attr:`input_tensor`, :attr:`alpha` or attr:`input_tensor_a`, attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::frac_bw,
        R"doc(Performs backward operations for frac on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::trunc_bw,
        R"doc(Performs backward operations for truncation on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::log_sigmoid_bw,
        R"doc(Performs backward operations for log sigmoid on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::fill_zero_bw,
        R"doc(Performs backward operations of fill zero on :attr:`input_tensor` with given :attr:`grad_tensor`. Returns an tensor of zeros like :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::i0_bw,
        R"doc(Performs backward operations for i0 on :attr:`input_tensor` or attr:`input_tensor_a` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::tan_bw,
        R"doc(Performs backward operations for tan on :attr:`input_tensor` or attr:`input_tensor_a` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::sigmoid_bw,
        R"doc(Performs backward operations for sigmoid on :attr:`input_tensor` or attr:`input_tensor_a` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::rsqrt_bw,
        R"doc(Performs backward operations for rsqrt on :attr:`input_tensor` or attr:`input_tensor_a` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::neg_bw,
        R"doc(Performs backward operations for neg on :attr:`input_tensor` or attr:`input_tensor_a` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::relu_bw,
        R"doc(Performs backward operations for relu on :attr:`input_tensor` or attr:`input_tensor_a` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::logit_bw,
        R"doc(Performs backward operations for logit on :attr:`input_tensor` or attr:`input_tensor_a` with given :attr:`grad_tensor`.)doc");
    
    detail::bind_unary_backward(
        module,
        ttnn::floor_bw,
        R"doc(Performs backward operations for floor on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
        module,
        ttnn::clamp_max_bw,
        R"doc(Performs backward operations for clamp max value on :attr:`input_tensor`, :attr:`max` with given :attr:`grad_tensor`.)doc");
    
    detail::bind_unary_backward(
        module,
        ttnn::hardshrink_bw,
        R"doc(Performs backward operations for hardshrink on :attr:`input_tensor`, :attr:`lambd` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::softshrink_bw,
        R"doc(Performs backward operations for softshrink on :attr:`input_tensor`, :attr:`lambd` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::leaky_relu_bw,
        R"doc(Performs backward operations for leaky relu on :attr:`input_tensor`, :attr:`negative_slope` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::elu_bw,
        R"doc(Performs backward operations for elu on :attr:`input_tensor`, :attr:`alpha` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::celu_bw,
        R"doc(Performs backward operations for celu on :attr:`input_tensor`, :attr:`alpha` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
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

    detail::bind_unary_backward(
        module,
        ttnn::relu6_bw,
        R"doc(Performs backward operations for relu6 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
        module,
        ttnn::abs_bw,
        R"doc(Performs backward operations for abs on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
        module,
        ttnn::silu_bw,
        R"doc(Performs backward operations for silu on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
        module,
        ttnn::selu_bw,
        R"doc(Performs backward operations for selu on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");
    
    detail::bind_unary_backward(
        module,
        ttnn::square_bw,
        R"doc(Performs backward operations for square on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
        module,
        ttnn::hardswish_bw,
        R"doc(Performs backward operations for  hardswish on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
        module,
        ttnn::tanhshrink_bw,
        R"doc(Performs backward operations for  tanhshrink on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
        module,
        ttnn::atanh_bw,
        R"doc(Performs backward operations for  atanh on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
        module,
        ttnn::asin_bw,
        R"doc(Performs backward operations for  asin on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
        module,
        ttnn::asinh_bw,
        R"doc(Performs backward operations for  asinh on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");
         
    detail::bind_unary_backward(
        module,
        ttnn::sin_bw,
        R"doc(Performs backward operations for sin on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
         module,
         ttnn::sinh_bw,
         R"doc(Performs backward operations for sinh on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
         module,
         ttnn::log10_bw,
         R"doc(Performs backward operations for log10 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
         module,
         ttnn::log1p_bw,
         R"doc(Performs backward operations for log1p on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward(
         module,
         ttnn::erfc_bw,
         R"doc(Performs backward operations for erfc on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");



}

}  // namespace binary_backward
}  // namespace operations
}  // namespace ttnn
