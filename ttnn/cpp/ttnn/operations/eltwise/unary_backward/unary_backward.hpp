
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
constexpr auto gt_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::GT_BW>>("ttnn::gt_bw");
constexpr auto lt_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LT_BW>>("ttnn::lt_bw");
constexpr auto le_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LE_BW>>("ttnn::le_bw");
constexpr auto ge_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::GE_BW>>("ttnn::ge_bw");
constexpr auto ne_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::NE_BW>>("ttnn::ne_bw");
constexpr auto lgamma_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LGAMMA_BW>>("ttnn::lgamma_bw");
constexpr auto fill_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::FILL_BW>>("ttnn::fill_bw");
constexpr auto hardsigmoid_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::HARDSIGMOID_BW>>("ttnn::hardsigmoid_bw");
constexpr auto cos_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::COS_BW>>("ttnn::cos_bw");
constexpr auto acosh_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ACOSH_BW>>("ttnn::acosh_bw");
constexpr auto acos_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ACOS_BW>>("ttnn::acos_bw");
constexpr auto atan_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ATAN_BW>>("ttnn::atan_bw");
constexpr auto rad2deg_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::RAD2DEG_BW>>("ttnn::rad2deg_bw");
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
constexpr auto selu_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SELU_BW>>("ttnn::selu_bw");
constexpr auto square_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SQUARE_BW>>("ttnn::square_bw");
constexpr auto hardswish_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::HARDSWISH_BW>>("ttnn::hardswish_bw");
constexpr auto tanhshrink_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::TANHSHRINK_BW>>("ttnn::tanhshrink_bw");
constexpr auto atanh_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ATANH_BW>>("ttnn::atanh_bw");
constexpr auto asin_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ASIN_BW>>("ttnn::asin_bw");
constexpr auto asinh_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ASINH_BW>>("ttnn::asinh_bw");
constexpr auto sin_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SIN_BW>>("ttnn::sin_bw");
constexpr auto sinh_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SINH_BW>>("ttnn::sinh_bw");
constexpr auto log10_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOG10_BW>>("ttnn::log10_bw");
constexpr auto log1p_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOG1P_BW>>("ttnn::log1p_bw");
constexpr auto erfc_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::ERFC_BW>>("ttnn::erfc_bw");
constexpr auto ceil_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::CEIL_BW>>("ttnn::ceil_bw");
constexpr auto softsign_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SOFTSIGN_BW>>("ttnn::softsign_bw");
constexpr auto cosh_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::COSH_BW>>("ttnn::cosh_bw");
constexpr auto logiteps_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOGITEPS_BW>>("ttnn::logiteps_bw");
constexpr auto log2_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::LOG2_BW>>("ttnn::log2_bw");
constexpr auto sign_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::SIGN_BW>>("ttnn::sign_bw");
constexpr auto fmod_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::FMOD_BW>>("ttnn::fmod_bw");
constexpr auto remainder_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::REMAINDER_BW>>("ttnn::remainder_bw");
constexpr auto div_no_nan_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::DIV_NO_NAN_BW>>("ttnn::div_no_nan_bw");
constexpr auto exp2_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::EXP2_BW>>("ttnn::exp2_bw");

}  // namespace ttnn
