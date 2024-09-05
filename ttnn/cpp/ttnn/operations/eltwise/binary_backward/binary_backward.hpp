
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/constants.hpp"
#include "device/binary_backward_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/complex_binary/device/complex_binary_op.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn {

namespace operations::binary_backward {

template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackwardTensor {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return OpHandler<binary_backward_op_type>::handle(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, output_memory_config);
    }
};

template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackwardOptionalFloatDefault {
    static std::vector<std::optional<Tensor>> invoke(
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
        return OpHandler<binary_backward_op_type>::handle(queue_id, grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, parameter, output_memory_config, are_required_outputs, input_a_grad, input_b_grad);
    }

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> input_a_grad = std::nullopt,
        std::optional<Tensor> input_b_grad = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return OpHandler<binary_backward_op_type>::handle(DefaultQueueId, grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, parameter, output_memory_config, are_required_outputs, input_a_grad, input_b_grad);
    }
};

template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackwardFloatDefault {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return OpHandler<binary_backward_op_type>::handle(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, parameter, output_memory_config);
    }
};

template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackwardIntDefault {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        int parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return OpHandler<binary_backward_op_type>::handle(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, parameter, output_memory_config);
    }
};

template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackwardFloat {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return OpHandler<binary_backward_op_type>::handle(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, parameter, output_memory_config);
    }
};

struct ExecuteBackwardMul  {

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float scalar,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const Tensor &other_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<ComplexTensor> invoke(
        const ComplexTensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_a_arg,
        const ComplexTensor &input_tensor_b_arg,
        const MemoryConfig &memory_config);
};

struct ExecuteBackwardBiasGelu {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        string approximate,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        float scalar,
        string approximate,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

};

struct ExecuteBackwardAdd {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float scalar,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<ComplexTensor> invoke(
        const ComplexTensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_a_arg,
        const ComplexTensor &input_tensor_b_arg,
        float alpha,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

};

struct ExecuteBackwardSub {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float scalar,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<ComplexTensor> invoke(
        const ComplexTensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_a_arg,
        const ComplexTensor &input_tensor_b_arg,
        float alpha,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

};

struct ExecuteBackwardDiv  {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float scalar,
        string round_mode = "None",
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const Tensor &other_tensor_arg,
        string round_mode = "None",
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<ComplexTensor> invoke(
        const ComplexTensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_a_arg,
        const ComplexTensor &input_tensor_b_arg,
        const MemoryConfig &memory_config);

};

struct ExecuteBackwardRemainder {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float scalar,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

};

struct ExecuteBackwardFmod {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float scalar,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

};


}  // operations::binary

constexpr auto atan2_bw = ttnn::register_operation<"ttnn::atan2_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::ATAN2_BW>>();
constexpr auto rsub_bw = ttnn::register_operation<"ttnn::rsub_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::RSUB_BW>>();
constexpr auto xlogy_bw = ttnn::register_operation<"ttnn::xlogy_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::XLOGY_BW>>();
constexpr auto hypot_bw = ttnn::register_operation<"ttnn::hypot_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::HYPOT_BW>>();
constexpr auto ldexp_bw = ttnn::register_operation<"ttnn::ldexp_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::LDEXP_BW>>();
constexpr auto logaddexp_bw = ttnn::register_operation<"ttnn::logaddexp_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::LOGADDEXP_BW>>();
constexpr auto logaddexp2_bw = ttnn::register_operation<"ttnn::logaddexp2_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::LOGADDEXP2_BW>>();
constexpr auto squared_difference_bw = ttnn::register_operation<"ttnn::squared_difference_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::SQUARED_DIFFERENCE_BW>>();
constexpr auto min_bw = ttnn::register_operation<"ttnn::min_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::MIN_BW>>();
constexpr auto max_bw = ttnn::register_operation<"ttnn::max_bw", operations::binary_backward::ExecuteBinaryBackwardTensor<operations::binary_backward::BinaryBackwardOpType::MAX_BW>>();

constexpr auto addalpha_bw = ttnn::register_operation<"ttnn::addalpha_bw", operations::binary_backward::ExecuteBinaryBackwardOptionalFloatDefault<operations::binary_backward::BinaryBackwardOpType::ADDALPHA_BW>>();

constexpr auto subalpha_bw = ttnn::register_operation<"ttnn::subalpha_bw", operations::binary_backward::ExecuteBinaryBackwardFloatDefault<operations::binary_backward::BinaryBackwardOpType::SUBALPHA_BW>>();

constexpr auto concat_bw = ttnn::register_operation<"ttnn::concat_bw", operations::binary_backward::ExecuteBinaryBackwardIntDefault<operations::binary_backward::BinaryBackwardOpType::CONCAT_BW>>();

constexpr auto mul_bw = ttnn::register_operation<"ttnn::mul_bw", operations::binary_backward::ExecuteBackwardMul>();

constexpr auto bias_gelu_bw = ttnn::register_operation<
    "ttnn::bias_gelu_bw",
    operations::binary_backward::ExecuteBackwardBiasGelu>();

constexpr auto add_bw = ttnn::register_operation<
    "ttnn::add_bw",
    operations::binary_backward::ExecuteBackwardAdd>();

constexpr auto sub_bw = ttnn::register_operation<
    "ttnn::sub_bw",
    operations::binary_backward::ExecuteBackwardSub>();

constexpr auto div_bw = ttnn::register_operation<"ttnn::div_bw", operations::binary_backward::ExecuteBackwardDiv>();

constexpr auto remainder_bw = ttnn::register_operation<
    "ttnn::remainder_bw",
    operations::binary_backward::ExecuteBackwardRemainder>();

constexpr auto fmod_bw = ttnn::register_operation<
    "ttnn::fmod_bw",
    operations::binary_backward::ExecuteBackwardFmod>();

}  // namespace ttnn
