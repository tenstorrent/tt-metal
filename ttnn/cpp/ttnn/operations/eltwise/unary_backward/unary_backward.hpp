
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

}  // namespace ttnn
