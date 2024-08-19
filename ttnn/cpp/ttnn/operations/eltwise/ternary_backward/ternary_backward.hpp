
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ternary_backward_op.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

namespace operations::ternary_backward {

template <TernaryBackwardOpType ternary_backward_op_type>
struct ExecuteTernaryBackward {

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor})),
                                            Tensor(operation::get_workers_for_op_output({input_tensor})),
                                            Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

     //Type 0: 3 inputs, 1 grad tensor, 1 float

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        const MemoryConfig &memory_config) {
        return OpHandler<ternary_backward_op_type>::handle(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, input_tensor_c_arg, memory_config);
    }
};

template <TernaryBackwardOpType ternary_backward_op_type>
struct ExecuteTernaryBackwardFloat {

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor})),
                                            Tensor(operation::get_workers_for_op_output({input_tensor})),
                                            Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Type 1: 3 inputs, 1 grad tensor, 1 float

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        float alpha,
        const MemoryConfig &memory_config) {
        return OpHandler<ternary_backward_op_type>::handle(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, input_tensor_c_arg, alpha, memory_config);
    }
};

template <TernaryBackwardOpType ternary_backward_op_type>
struct ExecuteTernaryBackwardOptional {

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor})),
                                            Tensor(operation::get_workers_for_op_output({input_tensor})),
                                            Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Q_ID, type1 args, optional output tensor for inputs based on are_required_outputs value

    static std::vector<OptionalTensor> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        OptionalTensor input_a_grad = std::nullopt,
        OptionalTensor input_b_grad = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return OpHandler<ternary_backward_op_type>::handle(queue_id, grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, input_tensor_c_arg, output_memory_config, are_required_outputs, input_a_grad, input_b_grad);
    }

    //type1 args, optional output tensor for inputs based on are_required_outputs value

    static std::vector<OptionalTensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        OptionalTensor input_a_grad = std::nullopt,
        OptionalTensor input_b_grad = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return OpHandler<ternary_backward_op_type>::handle(DefaultQueueId, grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, input_tensor_c_arg, output_memory_config, are_required_outputs, input_a_grad, input_b_grad);
    }
};

struct ExecuteTernaryBackwardLerp {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float scalar,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

};

}  // operations::ternary_backward

//type 1
constexpr auto addcmul_bw = ttnn::register_operation_with_auto_launch_op<
    "ttnn::addcmul_bw",
    operations::ternary_backward::ExecuteTernaryBackwardFloat<
        operations::ternary_backward::TernaryBackwardOpType::ADDCMUL_BW>>();
constexpr auto addcdiv_bw = ttnn::register_operation_with_auto_launch_op<
    "ttnn::addcdiv_bw",
    operations::ternary_backward::ExecuteTernaryBackwardFloat<
        operations::ternary_backward::TernaryBackwardOpType::ADDCDIV_BW>>();
constexpr auto where_bw = ttnn::register_operation<
    "ttnn::where_bw",
    operations::ternary_backward::ExecuteTernaryBackwardOptional<
        operations::ternary_backward::TernaryBackwardOpType::WHERE_BW>>();
constexpr auto lerp_bw = ttnn::register_operation<
    "ttnn::lerp_bw",
    operations::ternary_backward::ExecuteTernaryBackwardLerp>();
}  // namespace ttnn
