
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/binary_backward_op.cpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::binary_backward {

template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackward {

    static inline const std::array<TensorSchema, 3> input_tensor_schemas() {
        return {
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false}};
    }

    static inline std::vector<ttnn::Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor})),
                                            Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Type 1: 2 inputs, 1 grad tensor
    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b);
    }

    static std::vector<ttnn::Tensor> execute_on_worker_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const MemoryConfig &memory_config,
        const Tensor &input_tensor_b_arg) {

        auto op_type = utils::get_function_type1(binary_backward_op_type);
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, memory_config);
        }


    //Type 1: Type 1 with 1 float
    template <typename... Args>

    static std::vector<ttnn::Tensor> execute_on_worker_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        float alpha,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {

        auto op_type = utils::get_function_type1_w_float(binary_backward_op_type);
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, alpha, output_memory_config);
        }

    //Type 1: Type 1 with 1 string
    template <typename... Args>

    static std::vector<ttnn::Tensor> execute_on_worker_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        string value,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {

        auto op_type = utils::get_function_type1_w_string(binary_backward_op_type);
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, value, output_memory_config);
        }

    //Type 3 : Q_ID, type1 args, optional output tensor for inputs based on are_required_outputs value
    template <typename... Args>
    static auto input_tensors_to_validate(uint8_t queue_id, const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b);
    }

    static std::vector<std::optional<ttnn::Tensor>> execute_on_main_thread(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> optional_input_a_grad = std::nullopt,
        std::optional<Tensor> optional_input_b_grad = std::nullopt) {

        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        auto op_type = utils::get_function_type3(binary_backward_op_type);
        return op_type(queue_id, grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, output_memory_config, are_required_outputs, optional_input_a_grad, optional_input_b_grad);
    }

    //Type 3 : type1 args, optional output tensor for inputs based on are_required_outputs value
    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::vector<bool> are_required_outputs, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b);
    }

    static std::vector<std::optional<ttnn::Tensor>> execute_on_main_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> optional_input_a_grad = std::nullopt,
        std::optional<Tensor> optional_input_b_grad = std::nullopt) {

        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        auto op_type = utils::get_function_type3_wo_qid(binary_backward_op_type);
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, output_memory_config, are_required_outputs, optional_input_a_grad, optional_input_b_grad);
    }

    //Type 2 : Q_ID, type1 args, optional output tensor for inputs based on are_required_outputs value
    template <typename... Args>
    static auto input_tensors_to_validate(uint8_t queue_id, const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, float alpha, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b);
    }

    static std::vector<std::optional<ttnn::Tensor>> execute_on_main_thread(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float alpha,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> optional_input_a_grad = std::nullopt,
        std::optional<Tensor> optional_input_b_grad = std::nullopt) {

        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        auto op_type = utils::get_function_type2(binary_backward_op_type);
        return op_type(queue_id, grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, alpha, output_memory_config, are_required_outputs, optional_input_a_grad, optional_input_b_grad);
    }

    //Type 2 : type1 args, optional output tensor for inputs based on are_required_outputs value
    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, float alpha, std::vector<bool> are_required_outputs, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b);
    }

    static std::vector<std::optional<ttnn::Tensor>> execute_on_main_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float alpha,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> optional_input_a_grad = std::nullopt,
        std::optional<Tensor> optional_input_b_grad = std::nullopt) {

        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        auto op_type = utils::get_function_type2_wo_qid(binary_backward_op_type);
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, alpha, output_memory_config, are_required_outputs, optional_input_a_grad, optional_input_b_grad);
    }

};

}  // operations::binary

//type 1
constexpr auto atan2_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::ATAN2_BW>>("ttnn::atan2_bw");
constexpr auto embedding_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::EMBEDDING_BW>>("ttnn::embedding_bw");
constexpr auto subalpha_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::SUBALPHA_BW>>("ttnn::subalpha_bw");
constexpr auto sub_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::SUB_BW>>("ttnn::sub_bw");
constexpr auto xlogy_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::XLOGY_BW>>("ttnn::xlogy_bw");
constexpr auto hypot_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::HYPOT_BW>>("ttnn::hypot_bw");
constexpr auto ldexp_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::LDEXP_BW>>("ttnn::ldexp_bw");
constexpr auto logaddexp_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::LOGADDEXP_BW>>("ttnn::logaddexp_bw");
constexpr auto logaddexp2_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::LOGADDEXP2_BW>>("ttnn::logaddexp2_bw");
constexpr auto squared_difference_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::SQUARED_DIFFERENCE_BW>>("ttnn::squared_difference_bw");
constexpr auto binary_assign_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::BINARY_ASSIGN_BW>>("ttnn::binary_assign_bw");
constexpr auto concat_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::CONCAT_BW>>("ttnn::concat_bw");
constexpr auto binary_le_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::BINARY_LE_BW>>("ttnn::binary_le_bw");
constexpr auto rsub_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::RSUB_BW>>("ttnn::rsub_bw");
constexpr auto bias_gelu_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::BIAS_GELU_BW>>("ttnn::bias_gelu_bw");
constexpr auto binary_gt_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::BINARY_GT_BW>>("ttnn::binary_gt_bw");
constexpr auto binary_ne_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::BINARY_NE_BW>>("ttnn::binary_ne_bw");
constexpr auto binary_ge_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::BINARY_GE_BW>>("ttnn::binary_ge_bw");
constexpr auto min_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::MIN_BW>>("ttnn::min_bw");
constexpr auto max_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::MAX_BW>>("ttnn::max_bw");
constexpr auto div_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::DIV_BW>>("ttnn::div_bw");

//type 2
constexpr auto addalpha_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::ADDALPHA_BW>>("ttnn::addalpha_bw");

//type 3
constexpr auto add_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::ADD_BW>>("ttnn::add_bw");
constexpr auto binary_eq_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::BINARY_EQ_BW>>("ttnn::binary_eq_bw");

}  // namespace ttnn
