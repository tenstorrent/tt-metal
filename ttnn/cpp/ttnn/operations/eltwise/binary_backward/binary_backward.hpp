
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/binary_backward_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::binary_backward {

//OpHandler_binary_bw : get_function_binary_bw
template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackwardTensor {
    static std::vector<Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = get_function_binary_bw<binary_backward_op_type>();
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, output_memory_config);
    }
};

// OpHandler_binary_bw_opt_float_default : get_function_binary_bw_opt_float_default
template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackwardOptionalFloatDefault {
    static std::vector<std::optional<Tensor>> operator()(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> input_a_grad = std::nullopt,
        std::optional<Tensor> input_b_grad = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        auto op_type = get_function_binary_bw_opt_float_default<binary_backward_op_type>();
        return op_type(queue_id, grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, parameter, output_memory_config, are_required_outputs, input_a_grad, input_b_grad);
    }

    static std::vector<std::optional<Tensor>> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> input_a_grad = std::nullopt,
        std::optional<Tensor> input_b_grad = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        auto op_type = get_function_binary_bw_opt_float_default<binary_backward_op_type>();
        return op_type(DefaultQueueId, grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, parameter, output_memory_config, are_required_outputs, input_a_grad, input_b_grad);
    }
};

//OpHandler_binary_bw_float_default : get_function_binary_bw_float_default
template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackwardFloatDefault {
    static std::vector<Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = get_function_binary_bw_float_default<binary_backward_op_type>();
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, parameter, output_memory_config);
    }
};

// OpHandler_binary_bw_int_default : get_function_binary_bw_int_default
template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackwardIntDefault {
    static std::vector<Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        int parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = get_function_binary_bw_int_default<binary_backward_op_type>();
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, parameter, output_memory_config);
    }
};

//OpHandler_binary_bw_float : get_function_binary_bw_float
template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackwardFloat {
    static std::vector<Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = get_function_binary_bw_float<binary_backward_op_type>();
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, parameter, output_memory_config);
    }
};

template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackward {
    // Type 1: 2 inputs, 1 grad tensor

    static std::vector<ttnn::Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const std::optional<MemoryConfig> &memory_config,
        const Tensor &input_tensor_b_arg) {
        auto op_type = BinaryBackwardFunction::get_function_type1(binary_backward_op_type);
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, output_memory_config);
    }

        // Type 1: Type 1 with 1 string
        static std::vector<ttnn::Tensor> operator()(
            const Tensor &grad_tensor_arg,
            const Tensor &input_tensor_a_arg,
            string value,
            const Tensor &input_tensor_b_arg,
            const std::optional<MemoryConfig> &memory_config = std::nullopt) {
            auto op_type = BinaryBackwardFunction::get_function_type1_w_string(binary_backward_op_type);
            auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
            return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, value, output_memory_config);
        }

        // Type 3 : Q_ID, type1 args, optional output tensor for inputs based on are_required_outputs value

        static std::vector<std::optional<ttnn::Tensor>> operator()(
            uint8_t queue_id,
            const Tensor &grad_tensor_arg,
            const Tensor &input_tensor_a_arg,
            const Tensor &input_tensor_b_arg,
            const std::optional<MemoryConfig> &memory_config = std::nullopt,
            const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
            std::optional<Tensor> input_a_grad = std::nullopt,
            std::optional<Tensor> input_b_grad = std::nullopt) {
            auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
            auto op_type = BinaryBackwardFunction::get_function_type3(binary_backward_op_type);
            return op_type(
                queue_id,
                grad_tensor_arg,
                input_tensor_a_arg,
                input_tensor_b_arg,
                output_memory_config,
                are_required_outputs,
                input_a_grad,
                input_b_grad);
        }

        // Type 3 : type1 args, optional output tensor for inputs based on are_required_outputs value

        static std::vector<std::optional<ttnn::Tensor>> operator()(
            const Tensor &grad_tensor_arg,
            const Tensor &input_tensor_a_arg,
            const Tensor &input_tensor_b_arg,
            const std::optional<MemoryConfig> &memory_config = std::nullopt,
            const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
            std::optional<Tensor> input_a_grad = std::nullopt,
            std::optional<Tensor> input_b_grad = std::nullopt) {
            auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
            auto op_type = BinaryBackwardFunction::get_function_type3_wo_qid(binary_backward_op_type);
            return op_type(
                grad_tensor_arg,
                input_tensor_a_arg,
                input_tensor_b_arg,
                output_memory_config,
                are_required_outputs,
                input_a_grad,
                input_b_grad);
        }
};

}  // operations::binary

//OpHandler_binary_bw : get_function_binary_bw
constexpr auto atan2_bw = ttnn::register_operation<"ttnn::atan2_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::ATAN2_BW>>();
constexpr auto rsub_bw = ttnn::register_operation<"ttnn::rsub_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::RSUB_BW>>();
constexpr auto embedding_bw = ttnn::register_operation<"ttnn::embedding_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::EMBEDDING_BW>>();
constexpr auto xlogy_bw = ttnn::register_operation<"ttnn::xlogy_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::XLOGY_BW>>();
constexpr auto hypot_bw = ttnn::register_operation<"ttnn::hypot_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::HYPOT_BW>>();
constexpr auto ldexp_bw = ttnn::register_operation<"ttnn::ldexp_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::LDEXP_BW>>();
constexpr auto logaddexp_bw = ttnn::register_operation<"ttnn::logaddexp_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::LOGADDEXP_BW>>();

//OpHandler_binary_bw_opt_float_default : get_function_binary_bw_opt_float_default
constexpr auto addalpha_bw = ttnn::register_operation<"ttnn::addalpha_bw", operations::binary_backward::ExecuteBinaryBackwardOptionalFloatDefault<operations::binary_backward::BinaryBackwardOpType::ADDALPHA_BW>>();

//OpHandler_binary_bw_float_default : get_function_binary_bw_float_default
constexpr auto subalpha_bw = ttnn::register_operation<"ttnn::subalpha_bw", operations::binary_backward::ExecuteBinaryBackwardFloatDefault<operations::binary_backward::BinaryBackwardOpType::SUBALPHA_BW>>();

//OpHandler_binary_bw_int_default : get_function_binary_bw_int_default
constexpr auto concat_bw = ttnn::register_operation<"ttnn::concat_bw", operations::binary_backward::ExecuteBinaryBackwardIntDefault<operations::binary_backward::BinaryBackwardOpType::CONCAT_BW>>();

//OpHandler_binary_bw_float : get_function_binary_bw_float
constexpr auto lerp_bw = ttnn::register_operation<"ttnn::lerp_bw", operations::binary_backward::ExecuteBinaryBackwardFloat<operations::binary_backward::BinaryBackwardOpType::LERP_BW>>();

//type 1
constexpr auto logaddexp2_bw = ttnn::register_operation<"ttnn::logaddexp2_bw", operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::LOGADDEXP2_BW>>();
constexpr auto squared_difference_bw = ttnn::register_operation<"ttnn::squared_difference_bw", operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::SQUARED_DIFFERENCE_BW>>();
constexpr auto min_bw = ttnn::register_operation<"ttnn::min_bw", operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::MIN_BW>>();
constexpr auto max_bw = ttnn::register_operation<"ttnn::max_bw", operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::MAX_BW>>();

}  // namespace ttnn
