
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/unary_backward_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::unary_backward {

template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackward {

    static inline std::vector<ttnn::Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Type 1: 2 inputs, 1 grad tensor

    static std::vector<ttnn::Tensor> execute_on_worker_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {

        auto op_type = UnaryBackwardFunction::get_function_type1(unary_backward_op_type);
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, output_memory_config);
        }

    //Type 1: Type 1 with 1 float

    static std::vector<ttnn::Tensor> execute_on_worker_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float alpha,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {

        auto op_type = UnaryBackwardFunction::get_function_type1_w_float(unary_backward_op_type);
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, alpha, output_memory_config);
        }

    //Type 2: Type 1 with 2 float

    static std::vector<ttnn::Tensor> execute_on_worker_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float a,
        float b,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {

        auto op_type = UnaryBackwardFunction::get_function_type1_w_two_float(unary_backward_op_type);
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, a, b, output_memory_config);
        }

};

}  // operations::unary

constexpr auto mul_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::MUL_BW>>("ttnn::mul_bw");
constexpr auto clamp_min_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::CLAMP_MIN_BW>>("ttnn::clamp_min_bw");
constexpr auto clamp_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::CLAMP_BW>>("ttnn::clamp_bw");
constexpr auto assign_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ASSIGN_BW>>("ttnn::assign_bw");
constexpr auto multigammaln_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::MULTIGAMMALN_BW>>("ttnn::multigammaln_bw");
constexpr auto add_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ADD_BW>>("ttnn::add_bw");
constexpr auto eq_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::EQ_BW>>("ttnn::eq_bw");
constexpr auto lgamma_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LGAMMA_BW>>("ttnn::lgamma_bw");
constexpr auto sub_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SUB_BW>>("ttnn::sub_bw");
constexpr auto frac_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::FRAC_BW>>("ttnn::frac_bw");
constexpr auto trunc_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::TRUNC_BW>>("ttnn::trunc_bw");
constexpr auto log_sigmoid_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOG_SIGMOID_BW>>("ttnn::log_sigmoid_bw");
constexpr auto fill_zero_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::FILL_ZERO_BW>>("ttnn::fill_zero_bw");
constexpr auto i0_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::I0_BW>>("ttnn::i0_bw");
constexpr auto tan_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::TAN_BW>>("ttnn::tan_bw");
constexpr auto sigmoid_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SIGMOID_BW>>("ttnn::sigmoid_bw");
constexpr auto rsqrt_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::RSQRT_BW>>("ttnn::rsqrt_bw");
constexpr auto neg_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::NEG_BW>>("ttnn::neg_bw");
constexpr auto relu_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::RELU_BW>>("ttnn::relu_bw");
constexpr auto logit_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOGIT_BW>>("ttnn::logit_bw");
constexpr auto clamp_max_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::CLAMP_MAX_BW>>("ttnn::clamp_max_bw");
constexpr auto hardshrink_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::HARDSHRINK_BW>>("ttnn::hardshrink_bw");
constexpr auto softshrink_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SOFTSHRINK_BW>>("ttnn::softshrink_bw");
constexpr auto leaky_relu_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LEAKY_RELU_BW>>("ttnn::leaky_relu_bw");
constexpr auto elu_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ELU_BW>>("ttnn::elu_bw");
constexpr auto celu_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::CELU_BW>>("ttnn::celu_bw");
constexpr auto rpow_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::RPOW_BW>>("ttnn::rpow_bw");
constexpr auto floor_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::FLOOR_BW>>("ttnn::floor_bw");
constexpr auto round_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ROUND_BW>>("ttnn::round_bw");
constexpr auto log_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOG_BW>>("ttnn::log_bw");
constexpr auto relu6_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::RELU6_BW>>("ttnn::relu6_bw");
constexpr auto abs_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ABS_BW>>("ttnn::abs_bw");
constexpr auto silu_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SILU_BW>>("ttnn::silu_bw");


}  // namespace ttnn
