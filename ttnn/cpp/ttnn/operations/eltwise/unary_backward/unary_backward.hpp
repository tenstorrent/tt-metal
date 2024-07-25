
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/unary_backward_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::unary_backward {

//OpHandler_two_float : get_function_type1_w_two_float
template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardTwoFloat {

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Type 1: 1 inputs, 1 grad tensor, 2 float
    static std::vector<Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float min,
        float max,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = get_function_type1_w_two_float<unary_backward_op_type>();
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, min, max, output_memory_config);
    }
};

//OpHandler_two_float_with_default : get_function_type1_w_two_float_with_default
template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardTwoFloatWithDefault {
    //Type 1: 1 inputs, 1 grad tensor, 2 float
    static std::vector<Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float parameter_a,
        float parameter_b,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = get_function_type1_w_two_float_with_default<unary_backward_op_type>();
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, parameter_a, parameter_b, output_memory_config);
    }
};

//OpHandler_optional_float_params_with_default : get_function_optional_float_params_with_default
template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardOptionalFloatParamsWithDefault {

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Type 1: 1 inputs, 1 grad tensor, 2 float
    static std::vector<Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        std::optional<float> parameter_a,
        std::optional<float> parameter_b,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = get_function_optional_float_params_with_default<unary_backward_op_type>();
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, parameter_a, parameter_b, output_memory_config);
    }
};

//OpHandler_float_string_default : get_function_type1_float_string_default
template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardFloatStringDefault {

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Type 1: 1 inputs, 1 grad tensor, 1 float, 1 default string
    static std::vector<Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float parameter_a,
        string parameter_b,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = get_function_type1_float_string_default<unary_backward_op_type>();
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, parameter_a, parameter_b, output_memory_config);
    }
};

//OpHandler_string_default : get_function_type1_string_default
template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardStringDefault {

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Type 1: 1 inputs, 1 grad tensor, 1 float, 1 default string
    static std::vector<Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        string parameter_a,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = get_function_type1_string_default<unary_backward_op_type>();
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, parameter_a, output_memory_config);
    }
};

//OpHandler_shape : get_function_type1_shape
template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardShape {

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Type 1: 1 inputs, 1 grad tensor, 1 shape
    static std::vector<Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const tt::tt_metal::Shape &parameter_a,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = get_function_type1_shape<unary_backward_op_type>();
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, parameter_a, output_memory_config);
    }
};

//OpHandler_unary_optional_float : get_function_unary_optional_float
template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardOptionalFloat {

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Q_ID, type1 args, optional output tensor for input based on are_required_outputs value
    static std::vector<std::optional<Tensor>> operator()(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true},
        std::optional<Tensor> input_grad = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        auto op_type = get_function_unary_optional_float<unary_backward_op_type>();
        return op_type(queue_id, grad_tensor_arg, input_tensor_arg, parameter, output_memory_config, are_required_outputs, input_grad);
    }

    //type1 args, optional output tensor for inputs based on are_required_outputs value
    static std::vector<std::optional<Tensor>> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true},
        std::optional<Tensor> input_grad = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        auto op_type = get_function_unary_optional_float<unary_backward_op_type>();
        return op_type(DefaultQueueId, grad_tensor_arg, input_tensor_arg, parameter, output_memory_config, are_required_outputs, input_grad);
    }
};

//OpHandler_unary_optional : get_function_unary_optional
template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardOptional {

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Q_ID, type1 args, optional output tensor for input based on are_required_outputs value
    static std::vector<std::optional<Tensor>> operator()(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true},
        std::optional<Tensor> input_grad = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        auto op_type = get_function_unary_optional<unary_backward_op_type>();
        return op_type(queue_id, grad_tensor_arg, input_tensor_arg, output_memory_config, are_required_outputs, input_grad);
    }

    //type1 args, optional output tensor for inputs based on are_required_outputs value
    static std::vector<std::optional<Tensor>> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true},
        std::optional<Tensor> input_grad = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        auto op_type = get_function_unary_optional<unary_backward_op_type>();
        return op_type(DefaultQueueId, grad_tensor_arg, input_tensor_arg, output_memory_config, are_required_outputs, input_grad);
    }
};

//OpHandler_prod_bw : get_function_prod_bw
template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardProdBW {

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    static std::vector<Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        bool all_dimensions = true,
        int64_t dim = 0,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = get_function_prod_bw<unary_backward_op_type>();
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, all_dimensions, dim, output_memory_config);
    }
};

template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackward {

    static inline std::vector<ttnn::Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Type 1: 2 inputs, 1 grad tensor

    static std::vector<ttnn::Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = UnaryBackwardFunction::get_function_type1(unary_backward_op_type);
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, output_memory_config);
    }

    // Type 1: Type 1 with 1 float

    static std::vector<ttnn::Tensor> operator()(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float alpha,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto op_type = UnaryBackwardFunction::get_function_type1_w_float(unary_backward_op_type);
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, alpha, output_memory_config);
    }
};

}  // operations::unary

//ExecuteUnaryBackwardTwoFloat : get_function_type1_w_two_float
constexpr auto threshold_bw = ttnn::register_operation<
    "ttnn::threshold_bw",
    operations::unary_backward::ExecuteUnaryBackwardTwoFloat<
        operations::unary_backward::UnaryBackwardOpType::THRESHOLD_BW>>();

//OpHandler_optional_float_params_with_default : get_function_optional_float_params_with_default
constexpr auto clamp_bw = ttnn::register_operation<
    "ttnn::clamp_bw",
    operations::unary_backward::ExecuteUnaryBackwardOptionalFloatParamsWithDefault<
        operations::unary_backward::UnaryBackwardOpType::CLAMP_BW>>();

//ExecuteUnaryBackwardTwoFloatWithDefault : get_function_type1_w_two_float_with_default
constexpr auto softplus_bw = ttnn::register_operation<
    "ttnn::softplus_bw",
    operations::unary_backward::ExecuteUnaryBackwardTwoFloatWithDefault<
        operations::unary_backward::UnaryBackwardOpType::SOFTPLUS_BW>>();
constexpr auto hardtanh_bw = ttnn::register_operation<
    "ttnn::hardtanh_bw",
    operations::unary_backward::ExecuteUnaryBackwardTwoFloatWithDefault<
        operations::unary_backward::UnaryBackwardOpType::HARDTANH_BW>>();

//ExecuteUnaryBackwardFloatStringDefault : get_function_type1_float_string_default
constexpr auto div_bw = ttnn::register_operation<
    "ttnn::div_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloatStringDefault<
        operations::unary_backward::UnaryBackwardOpType::DIV_BW>>();
constexpr auto rdiv_bw = ttnn::register_operation<
    "ttnn::rdiv_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloatStringDefault<
        operations::unary_backward::UnaryBackwardOpType::RDIV_BW>>();
constexpr auto bias_gelu_bw = ttnn::register_operation<
    "ttnn::bias_gelu_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloatStringDefault<
        operations::unary_backward::UnaryBackwardOpType::BIAS_GELU_BW>>();

//ExecuteUnaryBackwardStringDefault : get_function_type1_string_default
constexpr auto gelu_bw = ttnn::register_operation<
    "ttnn::gelu_bw",
    operations::unary_backward::ExecuteUnaryBackwardStringDefault<
        operations::unary_backward::UnaryBackwardOpType::GELU_BW>>();

//ExecuteUnaryBackwardShape : get_function_type1_shape
constexpr auto repeat_bw = ttnn::register_operation<
    "ttnn::repeat_bw",
    operations::unary_backward::ExecuteUnaryBackwardShape<
        operations::unary_backward::UnaryBackwardOpType::REPEAT_BW>>();

//OpHandler_unary_optional_float : get_function_unary_optional_float
constexpr auto pow_bw = ttnn::register_operation<
    "ttnn::pow_bw",
    operations::unary_backward::ExecuteUnaryBackwardOptionalFloat<
        operations::unary_backward::UnaryBackwardOpType::POW_BW>>();

//OpHandler_unary_optional : get_function_unary_optional
constexpr auto exp_bw = ttnn::register_operation<
    "ttnn::exp_bw",
    operations::unary_backward::ExecuteUnaryBackwardOptional<
        operations::unary_backward::UnaryBackwardOpType::EXP_BW>>();
constexpr auto tanh_bw = ttnn::register_operation<
    "ttnn::tanh_bw",
    operations::unary_backward::ExecuteUnaryBackwardOptional<
        operations::unary_backward::UnaryBackwardOpType::TANH_BW>>();
constexpr auto sqrt_bw = ttnn::register_operation<
    "ttnn::sqrt_bw",
    operations::unary_backward::ExecuteUnaryBackwardOptional<
        operations::unary_backward::UnaryBackwardOpType::SQRT_BW>>();

//OpHandler_prod_bw : get_function_prod_bw
constexpr auto prod_bw = ttnn::register_operation<
    "ttnn::prod_bw",
    operations::unary_backward::ExecuteUnaryBackwardProdBW<operations::unary_backward::UnaryBackwardOpType::PROD_BW>>();

constexpr auto mul_bw = ttnn::register_operation<
    "ttnn::mul_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::MUL_BW>>();
constexpr auto assign_bw = ttnn::register_operation<
    "ttnn::assign_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ASSIGN_BW>>();
constexpr auto multigammaln_bw = ttnn::register_operation<
    "ttnn::multigammaln_bw",
    operations::unary_backward::ExecuteUnaryBackward<
        operations::unary_backward::UnaryBackwardOpType::MULTIGAMMALN_BW>>();
constexpr auto add_bw = ttnn::register_operation<
    "ttnn::add_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ADD_BW>>();
constexpr auto eq_bw = ttnn::register_operation<
    "ttnn::eq_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::EQ_BW>>();
constexpr auto gt_bw = ttnn::register_operation<
    "ttnn::gt_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::GT_BW>>();
constexpr auto lt_bw = ttnn::register_operation<
    "ttnn::lt_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LT_BW>>();
constexpr auto le_bw = ttnn::register_operation<
    "ttnn::le_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LE_BW>>();
constexpr auto ge_bw = ttnn::register_operation<
    "ttnn::ge_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::GE_BW>>();
constexpr auto ne_bw = ttnn::register_operation<
    "ttnn::ne_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::NE_BW>>();
constexpr auto lgamma_bw = ttnn::register_operation<
    "ttnn::lgamma_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LGAMMA_BW>>();
constexpr auto fill_bw = ttnn::register_operation<
    "ttnn::fill_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::FILL_BW>>();
constexpr auto hardsigmoid_bw = ttnn::register_operation<
    "ttnn::hardsigmoid_bw",
    operations::unary_backward::ExecuteUnaryBackward<
        operations::unary_backward::UnaryBackwardOpType::HARDSIGMOID_BW>>();
constexpr auto cos_bw = ttnn::register_operation<
    "ttnn::cos_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::COS_BW>>();
constexpr auto acosh_bw = ttnn::register_operation<
    "ttnn::acosh_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ACOSH_BW>>();
constexpr auto acos_bw = ttnn::register_operation<
    "ttnn::acos_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ACOS_BW>>();
constexpr auto atan_bw = ttnn::register_operation<
    "ttnn::atan_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ATAN_BW>>();
constexpr auto rad2deg_bw = ttnn::register_operation<
    "ttnn::rad2deg_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::RAD2DEG_BW>>();
constexpr auto sub_bw = ttnn::register_operation<
    "ttnn::sub_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SUB_BW>>();
constexpr auto frac_bw = ttnn::register_operation<
    "ttnn::frac_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::FRAC_BW>>();
constexpr auto trunc_bw = ttnn::register_operation<
    "ttnn::trunc_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::TRUNC_BW>>();
constexpr auto log_sigmoid_bw = ttnn::register_operation<
    "ttnn::log_sigmoid_bw",
    operations::unary_backward::ExecuteUnaryBackward<
        operations::unary_backward::UnaryBackwardOpType::LOG_SIGMOID_BW>>();
constexpr auto fill_zero_bw = ttnn::register_operation<
    "ttnn::fill_zero_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::FILL_ZERO_BW>>();
constexpr auto i0_bw = ttnn::register_operation<
    "ttnn::i0_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::I0_BW>>();
constexpr auto tan_bw = ttnn::register_operation<
    "ttnn::tan_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::TAN_BW>>();
constexpr auto sigmoid_bw = ttnn::register_operation<
    "ttnn::sigmoid_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SIGMOID_BW>>();
constexpr auto rsqrt_bw = ttnn::register_operation<
    "ttnn::rsqrt_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::RSQRT_BW>>();
constexpr auto neg_bw = ttnn::register_operation<
    "ttnn::neg_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::NEG_BW>>();
constexpr auto relu_bw = ttnn::register_operation<
    "ttnn::relu_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::RELU_BW>>();
constexpr auto logit_bw = ttnn::register_operation<
    "ttnn::logit_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOGIT_BW>>();
constexpr auto hardshrink_bw = ttnn::register_operation<
    "ttnn::hardshrink_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::HARDSHRINK_BW>>();
constexpr auto softshrink_bw = ttnn::register_operation<
    "ttnn::softshrink_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SOFTSHRINK_BW>>();
constexpr auto leaky_relu_bw = ttnn::register_operation<
    "ttnn::leaky_relu_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LEAKY_RELU_BW>>();
constexpr auto elu_bw = ttnn::register_operation<
    "ttnn::elu_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ELU_BW>>();
constexpr auto celu_bw = ttnn::register_operation<
    "ttnn::celu_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::CELU_BW>>();
constexpr auto rpow_bw = ttnn::register_operation<
    "ttnn::rpow_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::RPOW_BW>>();
constexpr auto floor_bw = ttnn::register_operation<
    "ttnn::floor_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::FLOOR_BW>>();
constexpr auto round_bw = ttnn::register_operation<
    "ttnn::round_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ROUND_BW>>();
constexpr auto log_bw = ttnn::register_operation<
    "ttnn::log_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOG_BW>>();
constexpr auto relu6_bw = ttnn::register_operation<
    "ttnn::relu6_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::RELU6_BW>>();
constexpr auto abs_bw = ttnn::register_operation<
    "ttnn::abs_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ABS_BW>>();
constexpr auto silu_bw = ttnn::register_operation<
    "ttnn::silu_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SILU_BW>>();
constexpr auto selu_bw = ttnn::register_operation<
    "ttnn::selu_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SELU_BW>>();
constexpr auto square_bw = ttnn::register_operation<
    "ttnn::square_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SQUARE_BW>>();
constexpr auto hardswish_bw = ttnn::register_operation<
    "ttnn::hardswish_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::HARDSWISH_BW>>();
constexpr auto tanhshrink_bw = ttnn::register_operation<
    "ttnn::tanhshrink_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::TANHSHRINK_BW>>();
constexpr auto atanh_bw = ttnn::register_operation<
    "ttnn::atanh_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ATANH_BW>>();
constexpr auto asin_bw = ttnn::register_operation<
    "ttnn::asin_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ASIN_BW>>();
constexpr auto asinh_bw = ttnn::register_operation<
    "ttnn::asinh_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ASINH_BW>>();
constexpr auto sin_bw = ttnn::register_operation<
    "ttnn::sin_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SIN_BW>>();
constexpr auto sinh_bw = ttnn::register_operation<
    "ttnn::sinh_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SINH_BW>>();
constexpr auto log10_bw = ttnn::register_operation<
    "ttnn::log10_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOG10_BW>>();
constexpr auto log1p_bw = ttnn::register_operation<
    "ttnn::log1p_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOG1P_BW>>();
constexpr auto erfc_bw = ttnn::register_operation<
    "ttnn::erfc_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ERFC_BW>>();
constexpr auto ceil_bw = ttnn::register_operation<
    "ttnn::ceil_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::CEIL_BW>>();
constexpr auto softsign_bw = ttnn::register_operation<
    "ttnn::softsign_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SOFTSIGN_BW>>();
constexpr auto cosh_bw = ttnn::register_operation<
    "ttnn::cosh_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::COSH_BW>>();
constexpr auto logiteps_bw = ttnn::register_operation<
    "ttnn::logiteps_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOGITEPS_BW>>();
constexpr auto log2_bw = ttnn::register_operation<
    "ttnn::log2_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOG2_BW>>();
constexpr auto sign_bw = ttnn::register_operation<
    "ttnn::sign_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SIGN_BW>>();
constexpr auto fmod_bw = ttnn::register_operation<
    "ttnn::fmod_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::FMOD_BW>>();
constexpr auto remainder_bw = ttnn::register_operation<
    "ttnn::remainder_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::REMAINDER_BW>>();
constexpr auto div_no_nan_bw = ttnn::register_operation<
    "ttnn::div_no_nan_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::DIV_NO_NAN_BW>>();
constexpr auto exp2_bw = ttnn::register_operation<
    "ttnn::exp2_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::EXP2_BW>>();
constexpr auto expm1_bw = ttnn::register_operation<
    "ttnn::expm1_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::EXPM1_BW>>();
constexpr auto reciprocal_bw = ttnn::register_operation<
    "ttnn::reciprocal_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::RECIPROCAL_BW>>();
constexpr auto digamma_bw = ttnn::register_operation<
    "ttnn::digamma_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::DIGAMMA_BW>>();
constexpr auto erfinv_bw = ttnn::register_operation<
    "ttnn::erfinv_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ERFINV_BW>>();
constexpr auto erf_bw = ttnn::register_operation<
    "ttnn::erf_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ERF_BW>>();
constexpr auto deg2rad_bw = ttnn::register_operation<
    "ttnn::deg2rad_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::DEG2RAD_BW>>();
constexpr auto polygamma_bw = ttnn::register_operation<
    "ttnn::polygamma_bw",
    operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::POLYGAMMA_BW>>();

}  // namespace ttnn
