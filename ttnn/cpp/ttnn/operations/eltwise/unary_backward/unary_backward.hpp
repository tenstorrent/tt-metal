
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/unary_backward_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn {

namespace operations::unary_backward {

struct ExecuteUnaryBackwardNeg {
    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardThreshold {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float min,
        float max,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardFloat {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float scalar,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return OpHandler<unary_backward_op_type>::handle(grad_tensor_arg, input_tensor_arg, scalar, output_memory_config);
        }

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return OpHandler<unary_backward_op_type>::handle(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, output_memory_config);
        }

};


template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardWoFloat {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return OpHandler<unary_backward_op_type>::handle(grad_tensor_arg, input_tensor_arg, output_memory_config);
        }

};

#define DEFINE_UNARY_BACKWARD_OPERATION_WITH_2_DEFAULT_FLOATS(op_name) \
struct ExecuteUnaryBackward##op_name { \
    static std::vector<Tensor> invoke( \
         const Tensor &grad_tensor_arg, \
        const Tensor &input_tensor_arg, \
        float parameter_a, \
        float parameter_b, \
        const std::optional<MemoryConfig> &memory_config = std::nullopt); \
};

template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardFloatWithDefault {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float parameter_a,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return OpHandler<unary_backward_op_type>::handle(grad_tensor_arg, input_tensor_arg, parameter_a, output_memory_config);
    }
};

template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardOp {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return OpHandler<unary_backward_op_type>::handle(grad_tensor_arg, input_tensor_arg, output_memory_config);
    }
};

struct ExecuteUnaryBackwardRsqrt {
    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardClamp {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        std::optional<float> parameter_a,
        std::optional<float> parameter_b,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardFloatStringDefault {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float parameter_a,
        string parameter_b,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return OpHandler<unary_backward_op_type>::handle(grad_tensor_arg, input_tensor_arg, parameter_a, parameter_b, output_memory_config);
    }
};

template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardStringDefault {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        string parameter_a,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return OpHandler<unary_backward_op_type>::handle(grad_tensor_arg, input_tensor_arg, parameter_a, output_memory_config);
    }
};

template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackwardShape {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const tt::tt_metal::LegacyShape &parameter_a,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return OpHandler<unary_backward_op_type>::handle(grad_tensor_arg, input_tensor_arg, parameter_a, output_memory_config);
    }
};

struct ExecuteUnaryBackwardPow {
    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float parameter,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardExp {
    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardTanh {
    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardSqrt {
    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardSilu {
    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardFill {
    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardProd {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        bool all_dimensions = true,
        int64_t dim = 0,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardRecip {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<ComplexTensor> invoke(
        const ComplexTensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_a_arg,
        const MemoryConfig &memory_config);

};

struct ExecuteUnaryBackwardAbs {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<ComplexTensor> invoke(
        const Tensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_a_arg,
        const MemoryConfig &memory_config);

};


struct ExecuteUnaryBackwardGelu{
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        string parameter_a,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        string parameter_a,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

};

DEFINE_UNARY_BACKWARD_OPERATION_WITH_2_DEFAULT_FLOATS(Softplus)
DEFINE_UNARY_BACKWARD_OPERATION_WITH_2_DEFAULT_FLOATS(Hardtanh)

}  // operations::unary

constexpr auto threshold_bw = ttnn::register_operation<
    "ttnn::threshold_bw",
    operations::unary_backward::ExecuteUnaryBackwardThreshold>();

constexpr auto multigammaln_bw = ttnn::register_operation<
    "ttnn::multigammaln_bw",
    operations::unary_backward::ExecuteUnaryBackwardWoFloat<
        operations::unary_backward::UnaryBackwardOpType::MULTIGAMMALN_BW>>();

constexpr auto lgamma_bw = ttnn::register_operation<
    "ttnn::lgamma_bw",
    operations::unary_backward::ExecuteUnaryBackwardWoFloat<
        operations::unary_backward::UnaryBackwardOpType::LGAMMA_BW>>();

constexpr auto fill_bw = ttnn::register_operation<
    "ttnn::fill_bw",
    operations::unary_backward::ExecuteUnaryBackwardFill>();

constexpr auto hardsigmoid_bw = ttnn::register_operation<
    "ttnn::hardsigmoid_bw",
    operations::unary_backward::ExecuteUnaryBackwardWoFloat<
        operations::unary_backward::UnaryBackwardOpType::HARDSIGMOID_BW>>();

constexpr auto cos_bw = ttnn::register_operation<
    "ttnn::cos_bw",
    operations::unary_backward::ExecuteUnaryBackwardWoFloat<
        operations::unary_backward::UnaryBackwardOpType::COS_BW>>();

constexpr auto acosh_bw = ttnn::register_operation<
    "ttnn::acosh_bw",
    operations::unary_backward::ExecuteUnaryBackwardWoFloat<
        operations::unary_backward::UnaryBackwardOpType::ACOSH_BW>>();

constexpr auto rpow_bw = ttnn::register_operation<
    "ttnn::rpow_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloat<
        operations::unary_backward::UnaryBackwardOpType::RPOW_BW>>();

constexpr auto div_no_nan_bw = ttnn::register_operation<
    "ttnn::div_no_nan_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloat<
        operations::unary_backward::UnaryBackwardOpType::DIV_NO_NAN_BW>>();

constexpr auto polygamma_bw = ttnn::register_operation<
    "ttnn::polygamma_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloat<
        operations::unary_backward::UnaryBackwardOpType::POLYGAMMA_BW>>();

//ExecuteUnaryBackwardOp : get_function_type1
constexpr auto acos_bw = ttnn::register_operation<
    "ttnn::acos_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::ACOS_BW>>();

constexpr auto atan_bw = ttnn::register_operation<
    "ttnn::atan_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::ATAN_BW>>();

constexpr auto rad2deg_bw = ttnn::register_operation<
    "ttnn::rad2deg_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::RAD2DEG_BW>>();

constexpr auto frac_bw = ttnn::register_operation<
    "ttnn::frac_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::FRAC_BW>>();

constexpr auto trunc_bw = ttnn::register_operation<
    "ttnn::trunc_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::TRUNC_BW>>();

constexpr auto log_sigmoid_bw = ttnn::register_operation<
    "ttnn::log_sigmoid_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::LOG_SIGMOID_BW>>();

constexpr auto fill_zero_bw = ttnn::register_operation<
    "ttnn::fill_zero_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::FILL_ZERO_BW>>();

constexpr auto i0_bw = ttnn::register_operation<
    "ttnn::i0_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::I0_BW>>();

constexpr auto relu6_bw = ttnn::register_operation<
    "ttnn::relu6_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::RELU6_BW>>();

constexpr auto selu_bw = ttnn::register_operation<
    "ttnn::selu_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::SELU_BW>>();

constexpr auto square_bw = ttnn::register_operation<
    "ttnn::square_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::SQUARE_BW>>();

constexpr auto tan_bw = ttnn::register_operation<
    "ttnn::tan_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::TAN_BW>>();

constexpr auto sigmoid_bw = ttnn::register_operation<
    "ttnn::sigmoid_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::SIGMOID_BW>>();

constexpr auto rsqrt_bw = ttnn::register_operation<
    "ttnn::rsqrt_bw",
    operations::unary_backward::ExecuteUnaryBackwardRsqrt>();

constexpr auto neg_bw = ttnn::register_operation<
    "ttnn::neg_bw",
    operations::unary_backward::ExecuteUnaryBackwardNeg>();

constexpr auto ceil_bw = ttnn::register_operation<
    "ttnn::ceil_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::CEIL_BW>>();

constexpr auto softsign_bw = ttnn::register_operation<
    "ttnn::softsign_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::SOFTSIGN_BW>>();

constexpr auto cosh_bw = ttnn::register_operation<
    "ttnn::cosh_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::COSH_BW>>();

constexpr auto log2_bw = ttnn::register_operation<
    "ttnn::log2_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::LOG2_BW>>();

constexpr auto sign_bw = ttnn::register_operation<
    "ttnn::sign_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::SIGN_BW>>();

constexpr auto exp2_bw = ttnn::register_operation<
    "ttnn::exp2_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::EXP2_BW>>();

constexpr auto expm1_bw = ttnn::register_operation<
    "ttnn::expm1_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::EXPM1_BW>>();

constexpr auto digamma_bw = ttnn::register_operation<
    "ttnn::digamma_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::DIGAMMA_BW>>();

constexpr auto erfinv_bw = ttnn::register_operation<
    "ttnn::erfinv_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::ERFINV_BW>>();

constexpr auto erf_bw = ttnn::register_operation<
    "ttnn::erf_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::ERF_BW>>();

constexpr auto deg2rad_bw = ttnn::register_operation<
    "ttnn::deg2rad_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::DEG2RAD_BW>>();

constexpr auto hardswish_bw = ttnn::register_operation<
    "ttnn::hardswish_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::HARDSWISH_BW>>();

constexpr auto tanhshrink_bw = ttnn::register_operation<
    "ttnn::tanhshrink_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::TANHSHRINK_BW>>();

constexpr auto atanh_bw = ttnn::register_operation<
    "ttnn::atanh_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::ATANH_BW>>();

constexpr auto asin_bw = ttnn::register_operation<
    "ttnn::asin_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::ASIN_BW>>();

constexpr auto asinh_bw = ttnn::register_operation<
    "ttnn::asinh_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::ASINH_BW>>();

constexpr auto sin_bw = ttnn::register_operation<
    "ttnn::sin_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::SIN_BW>>();

constexpr auto sinh_bw = ttnn::register_operation<
    "ttnn::sinh_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::SINH_BW>>();

constexpr auto log10_bw = ttnn::register_operation<
    "ttnn::log10_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::LOG10_BW>>();

constexpr auto log1p_bw = ttnn::register_operation<
    "ttnn::log1p_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::LOG1P_BW>>();

constexpr auto erfc_bw = ttnn::register_operation<
    "ttnn::erfc_bw",
    operations::unary_backward::ExecuteUnaryBackwardOp<
        operations::unary_backward::UnaryBackwardOpType::ERFC_BW>>();

constexpr auto hardshrink_bw = ttnn::register_operation<
    "ttnn::hardshrink_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloatWithDefault<
        operations::unary_backward::UnaryBackwardOpType::HARDSHRINK_BW>>();

constexpr auto softshrink_bw = ttnn::register_operation<
    "ttnn::softshrink_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloatWithDefault<
        operations::unary_backward::UnaryBackwardOpType::SOFTSHRINK_BW>>();

constexpr auto leaky_relu_bw = ttnn::register_operation<
    "ttnn::leaky_relu_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloatWithDefault<
        operations::unary_backward::UnaryBackwardOpType::LEAKY_RELU_BW>>();

constexpr auto elu_bw = ttnn::register_operation<
    "ttnn::elu_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloatWithDefault<
        operations::unary_backward::UnaryBackwardOpType::ELU_BW>>();

constexpr auto celu_bw = ttnn::register_operation<
    "ttnn::celu_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloatWithDefault<
        operations::unary_backward::UnaryBackwardOpType::CELU_BW>>();

constexpr auto logiteps_bw = ttnn::register_operation<
    "ttnn::logiteps_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloatWithDefault<
        operations::unary_backward::UnaryBackwardOpType::LOGITEPS_BW>>();

constexpr auto clamp_bw = ttnn::register_operation<
    "ttnn::clamp_bw",
    operations::unary_backward::ExecuteUnaryBackwardClamp>();

// Tensor + Float(Default) + Float(Default)
constexpr auto hardtanh_bw = ttnn::register_operation<"ttnn::hardtanh_bw", operations::unary_backward::ExecuteUnaryBackwardHardtanh>();
constexpr auto softplus_bw = ttnn::register_operation<"ttnn::softplus_bw", operations::unary_backward::ExecuteUnaryBackwardSoftplus>();

constexpr auto rdiv_bw = ttnn::register_operation<
    "ttnn::rdiv_bw",
    operations::unary_backward::ExecuteUnaryBackwardFloatStringDefault<
        operations::unary_backward::UnaryBackwardOpType::RDIV_BW>>();

constexpr auto gelu_bw = ttnn::register_operation<
    "ttnn::gelu_bw",
    operations::unary_backward::ExecuteUnaryBackwardGelu>();

constexpr auto repeat_bw = ttnn::register_operation<
    "ttnn::repeat_bw",
    operations::unary_backward::ExecuteUnaryBackwardShape<
        operations::unary_backward::UnaryBackwardOpType::REPEAT_BW>>();

constexpr auto pow_bw = ttnn::register_operation<
    "ttnn::pow_bw",
    operations::unary_backward::ExecuteUnaryBackwardPow>();

constexpr auto exp_bw = ttnn::register_operation<
    "ttnn::exp_bw",
    operations::unary_backward::ExecuteUnaryBackwardExp>();
constexpr auto tanh_bw = ttnn::register_operation<
    "ttnn::tanh_bw",
    operations::unary_backward::ExecuteUnaryBackwardTanh>();
constexpr auto sqrt_bw = ttnn::register_operation<
    "ttnn::sqrt_bw",
    operations::unary_backward::ExecuteUnaryBackwardSqrt>();

constexpr auto silu_bw = ttnn::register_operation<
    "ttnn::silu_bw",
    operations::unary_backward::ExecuteUnaryBackwardSilu>();

constexpr auto prod_bw = ttnn::register_operation<
    "ttnn::prod_bw",
    operations::unary_backward::ExecuteUnaryBackwardProd>();

constexpr auto relu_bw = ttnn::register_operation<
    "ttnn::relu_bw",
    operations::unary_backward::ExecuteUnaryBackwardWoFloat<operations::unary_backward::UnaryBackwardOpType::RELU_BW>>();
constexpr auto logit_bw = ttnn::register_operation<
    "ttnn::logit_bw",
    operations::unary_backward::ExecuteUnaryBackwardWoFloat<operations::unary_backward::UnaryBackwardOpType::LOGIT_BW>>();
constexpr auto floor_bw = ttnn::register_operation<
    "ttnn::floor_bw",
    operations::unary_backward::ExecuteUnaryBackwardWoFloat<operations::unary_backward::UnaryBackwardOpType::FLOOR_BW>>();
constexpr auto round_bw = ttnn::register_operation<
    "ttnn::round_bw",
    operations::unary_backward::ExecuteUnaryBackwardWoFloat<operations::unary_backward::UnaryBackwardOpType::ROUND_BW>>();
constexpr auto log_bw = ttnn::register_operation<
    "ttnn::log_bw",
    operations::unary_backward::ExecuteUnaryBackwardWoFloat<operations::unary_backward::UnaryBackwardOpType::LOG_BW>>();

// overload
constexpr auto reciprocal_bw = ttnn::register_operation<
    "ttnn::reciprocal_bw",
    operations::unary_backward::ExecuteUnaryBackwardRecip>();

constexpr auto abs_bw = ttnn::register_operation<
    "ttnn::abs_bw",
    operations::unary_backward::ExecuteUnaryBackwardAbs>();

}  // namespace ttnn
