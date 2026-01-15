// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"
#include "ttnn/operations/eltwise/complex_binary/device/complex_binary_op.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "tools/profiler/op_profiler.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/hal.hpp>

namespace ttnn::operations::unary_backward {

std::vector<Tensor> ExecuteUnaryBackwardClamp::invoke(
    const Tensor& grad,
    const Tensor& input,
    std::optional<float> min,
    std::optional<float> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(
        input.memory_config());  // TODO: Remove after ternary forward ops migration is completed
    TT_FATAL((max.has_value() || min.has_value()), "Only one of 'min' or 'max' can be None. Please provide one value");
    if (!max.has_value()) {
        Tensor minT = ttnn::ge(input, min.value(), std::nullopt, output_mem_config);
        Tensor result = ttnn::multiply(grad, minT, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    }
    if (!min.has_value()) {
        Tensor maxT = ttnn::le(input, max.value(), std::nullopt, output_mem_config);
        Tensor result = ttnn::multiply(grad, maxT, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    }
    Tensor minT = ttnn::ge(input, min.value(), std::nullopt, output_memory_config);
    Tensor maxT = ttnn::le(input, max.value(), std::nullopt, output_memory_config);
    Tensor result = ttnn::logical_and(minT, maxT, std::nullopt, output_memory_config);
    result = ttnn::multiply(grad, result, std::nullopt, output_memory_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardClamp::invoke(
    const Tensor& grad,
    const Tensor& input,
    std::optional<Tensor> min,
    std::optional<Tensor> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(
        input.memory_config());  // TODO: Remove after ternary forward ops migration is completed
    TT_FATAL((max.has_value() || min.has_value()), "Only one of 'min' or 'max' can be None. Please provide one value");
    if (!max.has_value()) {
        Tensor minT = ttnn::ge(input, min.value(), std::nullopt, output_mem_config);
        Tensor in_grad = ttnn::multiply(grad, minT, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(in_grad);
        return grad_tensor;
    }
    if (!min.has_value()) {
        Tensor maxT = ttnn::le(input, max.value(), std::nullopt, output_mem_config);
        Tensor in_grad = ttnn::multiply(grad, maxT, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(in_grad);
        return grad_tensor;
    }
    Tensor minT = ttnn::le(input, min.value(), std::nullopt, output_memory_config);
    Tensor maxT = ttnn::ge(input, max.value(), std::nullopt, output_memory_config);
    Tensor result = ttnn::logical_and(minT, maxT, std::nullopt, output_memory_config);
    result = ttnn::multiply(grad, result, std::nullopt, output_memory_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardClip::invoke(
    const Tensor& grad,
    const Tensor& input,
    std::optional<float> min,
    std::optional<float> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    return ExecuteUnaryBackwardClamp::invoke(grad, input, min, max, output_mem_config);
}

std::vector<Tensor> ExecuteUnaryBackwardClip::invoke(
    const Tensor& grad,
    const Tensor& input,
    std::optional<Tensor> min,
    std::optional<Tensor> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    return ExecuteUnaryBackwardClamp::invoke(grad, input, std::move(min), std::move(max), output_mem_config);
}

// Hardtanh
// result: torch.where((input <= min) | (input >= max), 0.0, grad)
std::vector<Tensor> ExecuteUnaryBackwardHardtanh::invoke(
    const Tensor& grad,
    const Tensor& input,
    float min,
    float max,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::where(
        ttnn::le(input, min, std::nullopt, output_mem_config),
        0.f,
        ttnn::where(ttnn::ge(input, max, std::nullopt, output_mem_config), 0.f, grad),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// threshold
// if input <= threshold = 0 else grad
std::vector<Tensor> ExecuteUnaryBackwardThreshold::invoke(
    const Tensor& grad,
    const Tensor& input,
    float threshold,
    float /*value*/,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::where(
        ttnn::gtz(ttnn::add(input, -threshold, std::nullopt, output_mem_config), output_mem_config),
        grad,
        0.f,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// Softplus
std::vector<Tensor> ExecuteUnaryBackwardSoftplus::invoke(
    const Tensor& grad,
    const Tensor& input,
    float beta,
    float threshold,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor mul_input_beta = ttnn::multiply(input, beta, std::nullopt, output_mem_config);
    Tensor exp_beta_self = ttnn::exp(mul_input_beta, false, output_mem_config);
    Tensor sub_result = ttnn::add(mul_input_beta, -threshold, std::nullopt, output_mem_config);
    Tensor temp = ttnn::multiply(
        ttnn::multiply(grad, exp_beta_self, std::nullopt, output_mem_config),
        ttnn::reciprocal(ttnn::add(exp_beta_self, 1.0f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    Tensor grad_result = ttnn::where(ttnn::gtz(sub_result, output_mem_config), grad, temp, output_mem_config);
    mul_input_beta.deallocate();
    exp_beta_self.deallocate();
    sub_result.deallocate();
    temp.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardRdiv::invoke(
    const Tensor& grad,
    const Tensor& input,
    float scalar,
    const std::optional<std::string>& rounding_mode,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    TT_FATAL(
        (rounding_mode == std::nullopt || rounding_mode == "trunc" || rounding_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    if (rounding_mode == std::nullopt) {
        Tensor result = ttnn::where(
            ttnn::nez(input),
            ttnn::multiply(
                ttnn::neg(grad, output_mem_config),
                (ttnn::multiply(
                    ttnn::reciprocal(ttnn::square(input, output_mem_config)), scalar, std::nullopt, output_mem_config)),
                std::nullopt,
                output_mem_config),
            t_nan,
            output_mem_config);
        if (scalar > 0) {
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config),
                    ttnn::ltz(grad, output_mem_config),
                    std::nullopt,
                    output_mem_config),
                t_inf,
                result,
                output_mem_config);
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config),
                    ttnn::gtz(grad, output_mem_config),
                    std::nullopt,
                    output_mem_config),
                -t_inf,
                result,
                output_mem_config);
        } else if (scalar < 0) {
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config),
                    ttnn::ltz(grad, output_mem_config),
                    std::nullopt,
                    output_mem_config),
                -t_inf,
                result,
                output_mem_config);
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config),
                    ttnn::gtz(grad, output_mem_config),
                    std::nullopt,
                    output_mem_config),
                t_inf,
                result,
                output_mem_config);
        }
        grad_tensor.emplace_back(result);
    } else {
        Tensor result = ttnn::zeros_like(grad, grad.dtype(), grad.layout(), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(result);
    }
    return grad_tensor;
}

// unary_pow:
// grad_input = grad * exponent * torch.pow(input, exponent - 1)
std::vector<std::optional<Tensor>> ExecuteUnaryBackwardPow::invoke(
    const Tensor& grad,
    const Tensor& input,
    float exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;
    input_grad = input_grad.value_or(ttnn::empty_like(input));
    const float ZERO_THRESHOLD = std::numeric_limits<float>::epsilon() * 10.0f;
    TT_FATAL(exponent >= 0.0, "negative exponents are not supported; use recip(pow(input,abs(exponent)))");
    if (std::abs(exponent) < ZERO_THRESHOLD) {
        input_grad = ttnn::zeros_like(input);
        grad_tensor.emplace_back(input_grad);
        return grad_tensor;
    }

    Tensor power_input = ttnn::pow(input, std::fabs(exponent - 1.0f), output_mem_config);
    if (exponent < 1.0f) {
        power_input = ttnn::reciprocal(power_input, output_mem_config);
    }

    Tensor result = ttnn::multiply(power_input, exponent, std::nullopt, output_mem_config);
    power_input.deallocate();
    Tensor final_result = ttnn::multiply(result, grad, std::nullopt, output_mem_config);
    result.deallocate();
    // Handle negative inputs by returning infinity
    where(ttnn::lez(input), std::numeric_limits<float>::infinity(), final_result, output_mem_config, input_grad);
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardExp::invoke(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;

    input_grad = input_grad.value_or(ttnn::empty_like(input));
    Tensor exp_result = ttnn::exp(input, false, output_mem_config);
    Tensor result = ttnn::multiply(grad, exp_result, std::nullopt, output_mem_config, input_grad);
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardTanh::invoke(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;

    input_grad = input_grad.value_or(ttnn::empty_like(input));
    Tensor tanh_res = ttnn::tanh(input, output_mem_config);
    tanh_res = ttnn::square(tanh_res, output_mem_config);
    tanh_res = ttnn::rsub(tanh_res, 1.0f, std::nullopt, output_mem_config);
    ttnn::multiply(grad, tanh_res, std::nullopt, output_mem_config, input_grad);
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardSqrt::invoke(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;

    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();

    input_grad = input_grad.value_or(ttnn::empty_like(input));
    ttnn::sqrt(input, false, output_mem_config, input_grad);
    ttnn::multiply(
        grad,
        ttnn::reciprocal(ttnn::multiply(input_grad.value(), 2.0, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config,
        input_grad);
    where(ttnn::lez(input, output_mem_config), t_nan, input_grad.value(), output_mem_config, input_grad);
    where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
        -t_inf,
        input_grad.value(),
        output_mem_config,
        input_grad);
    where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_inf,
        input_grad.value(),
        output_mem_config,
        input_grad);
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardMultigammaln::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor digamma_result =
        ttnn::multiply(grad, ttnn::digamma(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor digamma_result_2 = ttnn::multiply(
        grad,
        ttnn::digamma(ttnn::add(input, -0.5, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);

    Tensor grad_result = ttnn::add(digamma_result, digamma_result_2, std::nullopt, output_mem_config);

    digamma_result = ttnn::multiply(
        grad,
        ttnn::digamma(ttnn::add(input, -1.0, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = ttnn::add(grad_result, digamma_result, std::nullopt, output_mem_config);

    digamma_result = ttnn::multiply(
        grad,
        ttnn::digamma(ttnn::add(input, -1.5, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = ttnn::add(grad_result, digamma_result, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardLgamma::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    auto output_memory_config = output_mem_config.value_or(
        input.memory_config());  // TODO: Remove after ternary forward ops migration is completed
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(grad, ttnn::digamma(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardFrac::invoke(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardTrunc::invoke(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::zeros_like(grad, grad.dtype(), grad.layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// return: grad_output * (max_deriv - sign * (z / (1 + z)))
// z = exp(-abs(input))
std::vector<Tensor> ExecuteUnaryBackwardLogSigmoid::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor max_deriv = ttnn::where(ttnn::ltz(input, output_mem_config), 1.f, 0.f, output_mem_config);
    Tensor in_sign = ttnn::where(ttnn::ltz(input, output_mem_config), 1.f, -1.f, output_mem_config);
    Tensor in_abs = ttnn::abs(input, output_mem_config);
    Tensor z = ttnn::exp(ttnn::neg(in_abs, output_mem_config), false, output_mem_config);

    Tensor mul_z = ttnn::multiply(
        z,
        ttnn::reciprocal((ttnn::add(z, 1.0f, std::nullopt, output_mem_config)), output_mem_config),
        std::nullopt,
        output_mem_config);

    Tensor mul_sign = ttnn::multiply(in_sign, mul_z, std::nullopt, output_mem_config);
    Tensor sub_max = ttnn::subtract(max_deriv, mul_sign, std::nullopt, output_mem_config);

    Tensor grad_result = ttnn::multiply(grad, sub_max, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardFillZero::invoke(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::zeros_like(grad, grad.dtype(), grad.layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

//   name: i0(Tensor self) -> Tensor
//   self: grad * at::special_i1(self)
//   result: auto_element_wise
std::vector<Tensor> ExecuteUnaryBackwardI0::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor i1_input = ttnn::i1(input, output_mem_config);
    Tensor result = ttnn::multiply(grad, i1_input, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardTan::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tan_result = ttnn::tan(input, output_mem_config);
    Tensor result = ttnn::multiply(
        grad,
        ttnn::add(ttnn::square(tan_result, output_mem_config), 1.0f, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// grad(sigmoid) = grad*(1 - sigmoid(x))*sigmoid(x)
std::vector<Tensor> ExecuteUnaryBackwardSigmoid::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    bool approximate_mode = false;
    Tensor sig_result = ttnn::sigmoid(input, (int)unary::VecMode::RC, approximate_mode, output_mem_config);
    Tensor rsub_term = ttnn::rsub(sig_result, 1.0f, std::nullopt, output_mem_config);
    Tensor prod_term_1 = ttnn::multiply(sig_result, rsub_term, std::nullopt, output_mem_config);
    Tensor prod_term_2 = ttnn::multiply(prod_term_1, grad, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(prod_term_2);
    return grad_tensor;
}

std::vector<std::optional<ttnn::Tensor>> ExecuteUnaryBackwardRsqrt::invoke(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result;
    if (!input_grad.has_value()) {
        input_grad = ttnn::empty_like(grad);
    }
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");

    ttnn::rsqrt(input, false, output_mem_config, input_grad);
    ttnn::power(input_grad.value(), 3, output_mem_config, input_grad);
    ttnn::multiply(
        ttnn::multiply(grad, input_grad.value(), std::nullopt, output_mem_config),
        -0.5,
        std::nullopt,
        output_mem_config,
        input_grad);
    where(ttnn::eqz(input, output_mem_config), t_inf, input_grad.value(), output_mem_config, input_grad);
    where(ttnn::ltz(input, output_mem_config), t_nan, input_grad.value(), output_mem_config, input_grad);
    where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_nan,
        input_grad.value(),
        output_mem_config,
        input_grad);

    result.emplace_back(input_grad);
    return result;
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardNeg::invoke(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt};
    input_grad = input_grad.value_or(ttnn::empty_like(input));
    result[0] = ttnn::neg(grad, output_mem_config, input_grad);
    return result;
}

std::vector<Tensor> ExecuteUnaryBackwardRelu::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(ttnn::gtz(input, output_mem_config), grad, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// fill_bw:
// name: fill.Scalar(Tensor self, Scalar value) -> Tensor
// self: zeros_like(grad)
// result: at::fill(self_t, 0)
std::vector<std::optional<Tensor>> ExecuteUnaryBackwardFill::invoke(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    std::vector<std::optional<Tensor>> result = {std::nullopt};
    result[0] = input_grad.has_value()
                    ? ttnn::zeros_like(grad, std::nullopt, std::nullopt, std::nullopt, std::nullopt, input_grad)
                    : ttnn::zeros_like(grad);
    return result;
}

std::vector<Tensor> ExecuteUnaryBackwardHardsigmoid::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = ttnn::where(
        ttnn::logical_or(
            ttnn::le(input, -3, std::nullopt, output_mem_config),
            ttnn::ge(input, 3, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        0.f,
        ttnn::multiply(grad, 1.0 / 6),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// name: cos(Tensor self) -> Tensor
// self: grad * -self.sin()
std::vector<Tensor> ExecuteUnaryBackwardCos::invoke(
    const Tensor& grad, const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        grad,
        (ttnn::neg(ttnn::sin(input_tensor, output_mem_config), output_mem_config)),
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardAcosh::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor in_sq = ttnn::square(input, output_mem_config);
    Tensor in_rsqrt =
        ttnn::rsqrt(ttnn::subtract(in_sq, 1.0, std::nullopt, output_mem_config), false, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, in_rsqrt, std::nullopt, output_mem_config);
    float t_nan = tt::tt_metal::hal::get_nan();
    float t_inf = tt::tt_metal::hal::get_inf();

    Tensor check_condition =
        ttnn::multiply(ttnn::signbit(grad, output_mem_config), -1.0f, std::nullopt, output_mem_config);

    grad_a = ttnn::where(
        ttnn::logical_or(
            ttnn::lt(in_sq, 1.0f, std::nullopt, output_mem_config),
            ttnn::logical_and(
                ttnn::eq(input, 1.0f, std::nullopt, output_mem_config),
                ttnn::eqz(grad, output_mem_config),
                std::nullopt,
                output_mem_config),
            std::nullopt,
            output_mem_config),
        t_nan,
        ttnn::where(
            ttnn::logical_and(
                ttnn::le(input, 1.0f, std::nullopt, output_mem_config),
                ttnn::ge(input, -1.0f, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            ttnn::multiply(
                ttnn::add(
                    check_condition, ttnn::eqz(check_condition, output_mem_config), std::nullopt, output_mem_config),
                t_inf,
                std::nullopt,
                output_mem_config),
            grad_a,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// # - name: acos(Tensor self) -> Tensor
// #   self: grad * -((-self * self + 1).rsqrt())
std::vector<Tensor> ExecuteUnaryBackwardAcos::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor neg_in = ttnn::neg(input, output_mem_config);
    Tensor in_rsqrt = ttnn::rsqrt(
        ttnn::add(
            ttnn::multiply(neg_in, input, std::nullopt, output_mem_config), 1.0f, std::nullopt, output_mem_config),
        false,
        output_mem_config);
    in_rsqrt = ttnn::neg(in_rsqrt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, in_rsqrt, std::nullopt, output_mem_config);
    Tensor t_inf = ttnn::multiply(
        ttnn::sign(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_or(
            ttnn::lt(input, -1.0f, std::nullopt, output_mem_config),
            ttnn::gt(input, 1.0f, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        std::nanf(" "),
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::eq(input, -1.0f, std::nullopt, output_mem_config),
        t_inf,
        where(ttnn::eq(input, 1.0f, std::nullopt, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardAtan::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::SQUARE},
        EltwiseUnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
        EltwiseUnaryWithParam{UnaryOpType::RECIP}};
    Tensor grad_a =
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardRad2deg::invoke(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float M_180_PI = 180 / M_PI;
    Tensor grad_result = ttnn::multiply(grad, M_180_PI, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardLogit::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(
        grad,
        ttnn::reciprocal(ttnn::multiply(
            input, ttnn::rsub(input, 1.0f, std::nullopt, output_mem_config), std::nullopt, output_mem_config)),
        std::nullopt,
        output_mem_config);
    Tensor status = ttnn::logical_and(
        ttnn::ge(input, 0.0f, std::nullopt, output_mem_config),
        ttnn::le(input, 1.0f, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = where(ttnn::eq(status, 1.0f, std::nullopt, output_mem_config), grad_result, std::nanf(""));
    grad_result = where(
        ttnn::logical_or(
            ttnn::eq(input, 0.0f, std::nullopt, output_mem_config),
            ttnn::eq(input, 1.0f, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        ttnn::multiply(
            ttnn::sign(grad, output_mem_config),
            std::numeric_limits<float>::infinity(),
            std::nullopt,
            output_mem_config),
        grad_result,
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
// square
// result:  2 * input * grad_data
std::vector<Tensor> ExecuteUnaryBackwardSquare::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(
        ttnn::multiply(grad, 2.0f, std::nullopt, output_mem_config), input, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardHardshrink::invoke(
    const Tensor& grad, const Tensor& input_tensor, float lambd, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor hardshrink_result = ttnn::hardshrink(input_tensor, lambd, output_mem_config);
    Tensor result = where(ttnn::eqz(hardshrink_result, output_mem_config), 0.0f, grad, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// softshrink
//  result: torch.where(self < -lambd, grad, torch.where(self > lambd, grad, torch.tensor(0.0)))
std::vector<Tensor> ExecuteUnaryBackwardSoftshrink::invoke(
    const Tensor& grad, const Tensor& input_tensor, float lambd, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::where(
        ttnn::logical_or(
            ttnn::lt(input_tensor, -lambd, std::nullopt, output_mem_config),
            ttnn::gt(input_tensor, lambd, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        grad,
        0.f,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// Leaky_Relu
// result: torch.where(self > 0, grad_output, grad_output * negative_slope)
std::vector<Tensor> ExecuteUnaryBackwardLeakyRelu::invoke(
    const Tensor& grad,
    const Tensor& input,
    float negative_slope,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config),
        grad,
        ttnn::multiply(grad, negative_slope, std::nullopt, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// ELU
// result : grad * (torch.where(input > 0, 1, alpha * torch.exp(input)))
std::vector<Tensor> ExecuteUnaryBackwardElu::invoke(
    const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config),
        grad,
        ttnn::multiply(
            grad,
            ttnn::multiply(ttnn::exp(input, false, output_mem_config), alpha, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// Celu
// result: torch.where((input > 0), grad, grad * torch.exp(input / alpha))
std::vector<Tensor> ExecuteUnaryBackwardCelu::invoke(
    const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float div_alpha = (1.0 / alpha);
    Tensor div_result = ttnn::multiply(input, div_alpha, std::nullopt, output_mem_config);
    Tensor exp_result = ttnn::exp(div_result, false, output_mem_config);
    Tensor grad_result = where(
        ttnn::gt(input, 0.0, std::nullopt, output_mem_config),
        grad,
        ttnn::multiply(grad, exp_result, std::nullopt, output_mem_config),
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardRpow::invoke(
    const Tensor& grad, const Tensor& input, float exponent, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    Tensor grad_result = ttnn::zeros_like(input, input.dtype(), input.layout(), std::nullopt, output_mem_config);
    if (exponent != 0.0) {
        grad_result = ttnn::multiply(
            grad,
            ttnn::multiply(pow(input, exponent - 1, output_mem_config), exponent, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config);
        grad_result = ttnn::where(ltz(input, output_mem_config), t_nan, grad_result, output_mem_config);
    }
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardFloor::invoke(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = ttnn::zeros_like(grad);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardRound::invoke(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = ttnn::zeros_like(grad);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardLog::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(where(
        ttnn::eqz(input, output_mem_config),
        where(
            ttnn::eqz(grad, output_mem_config),
            std::nanf(""),
            ttnn::multiply(
                ttnn::sign(grad, output_mem_config),
                std::numeric_limits<float>::infinity(),
                std::nullopt,
                output_mem_config),
            output_mem_config),
        grad_a,
        output_mem_config));
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardRelu6::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(ttnn::le(input, 0.0f, std::nullopt, output_mem_config), 0.0f, 6.0f, output_mem_config);
    grad_result = where(
        ttnn::logical_and(
            ttnn::gtz(input, output_mem_config),
            ttnn::lt(input, 6.0f, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        grad,
        grad_result,
        output_mem_config);
    grad_result = where(ttnn::ge(input, 6.0f, std::nullopt, output_mem_config), 0.0f, grad_result, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _abs_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(grad, ttnn::sign(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// Silu
// result:  grad * sigmoid_result * (1 + input * (1 - sigmoid_result))
std::vector<std::optional<Tensor>> ExecuteUnaryBackwardSilu::invoke(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt};

    input_grad = input_grad.value_or(ttnn::empty_like(input));
    bool approximate_mode = false;
    Tensor sigmoid_res = ttnn::sigmoid(input, (int)unary::VecMode::RC, approximate_mode, output_mem_config);
    Tensor grad_sigmoid = ttnn::multiply(grad, sigmoid_res, std::nullopt, output_mem_config);
    Tensor add_sub = ttnn::add(
        ttnn::multiply(
            ttnn::rsub(sigmoid_res, 1.0f, std::nullopt, output_mem_config), input, std::nullopt, output_mem_config),
        1.0f,
        std::nullopt,
        output_mem_config);
    ttnn::multiply(grad_sigmoid, add_sub, std::nullopt, output_mem_config, input_grad);

    result[0] = input_grad;
    return result;
}

// Selu
// result:  torch.where(input > 0, grad * lambd, grad * lambd * alpha * torch.exp(input))
std::vector<Tensor> ExecuteUnaryBackwardSelu::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_lambd = ttnn::multiply(grad, 1.0507f, std::nullopt, output_mem_config);
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config),
        grad_lambd,
        ttnn::multiply(
            ttnn::multiply(grad_lambd, 1.673260f, std::nullopt, output_mem_config),
            ttnn::exp(input, false, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// Hardswish
// result: torch.where(input < -3,0.0,torch.where(input <= 3, grad * ((input / 3) + 0.5), grad),)
std::vector<Tensor> ExecuteUnaryBackwardHardswish::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::lt(input, -3.0f, std::nullopt, output_mem_config),
        0.f,
        where(
            ttnn::le(input, 3.0f, std::nullopt, output_mem_config),
            ttnn::multiply(
                grad,
                ttnn::add(
                    ttnn::multiply(input, 0.3333f, std::nullopt, output_mem_config),
                    0.5f,
                    std::nullopt,
                    output_mem_config),
                std::nullopt,
                output_mem_config),
            grad),
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// tanhshrink
// result:  torch.square(torch.tanh(input)) * grad_data
std::vector<Tensor> ExecuteUnaryBackwardTanhshrink::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tanh_res = ttnn::square(ttnn::tanh(input, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(ttnn::multiply(grad, tanh_res, std::nullopt, output_mem_config));
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardAtanh::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::SQUARE},
        EltwiseUnaryWithParam{UnaryOpType::SUB_UNARY_SFPU, 1.0f},
        EltwiseUnaryWithParam{UnaryOpType::NEG},
        EltwiseUnaryWithParam{UnaryOpType::RECIP}};

    Tensor grad_a =
        ttnn::multiply(grad, unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(ttnn::eqz(grad, output_mem_config), t_nan, grad_a, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(grad, output_mem_config), ttnn::eqz(input, output_mem_config)),
        0.f,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::logical_or(
                ttnn::eq(input, 1, std::nullopt, output_mem_config),
                ttnn::eq(input, -1, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            ttnn::nez(grad, output_mem_config)),
        t_inf,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eq(grad_a, t_inf, std::nullopt, output_mem_config), ttnn::ltz(grad, output_mem_config)),
        -t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// Asin
// result: grad * (-self * self + 1).rsqrt()
std::vector<Tensor> ExecuteUnaryBackwardAsin::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::SQUARE},
        EltwiseUnaryWithParam{UnaryOpType::NEG},
        EltwiseUnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
        EltwiseUnaryWithParam{UnaryOpType::RSQRT}};

    Tensor grad_result =
        ttnn::multiply(grad, unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");
    Tensor sub_one = ttnn::add(input, -1, std::nullopt, output_mem_config);
    Tensor sub_minus_one = ttnn::add(input, 1, std::nullopt, output_mem_config);
    Tensor result = where(
        ttnn::ltz(sub_minus_one, output_mem_config),
        t_nan,
        where(
            ttnn::gtz(sub_one, output_mem_config),
            t_nan,
            where(
                ttnn::eqz(sub_minus_one, output_mem_config),
                ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                where(
                    ttnn::eqz(sub_one, output_mem_config),
                    ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                    grad_result,
                    output_mem_config),
                output_mem_config),
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// Asinh
// result: grad * (self * self + 1).rsqrt()
std::vector<Tensor> ExecuteUnaryBackwardAsinh::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::SQUARE},
        EltwiseUnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
        EltwiseUnaryWithParam{UnaryOpType::RSQRT}};
    Tensor grad_result =
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// name: sin(Tensor self) -> Tensor
// self: grad * self.cos()
std::vector<Tensor> ExecuteUnaryBackwardSin::invoke(
    const Tensor& grad, const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_input =
        ttnn::multiply(grad, ttnn::cos(input_tensor, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_input);
    return grad_tensor;
}

// name: sinh(Tensor self) -> Tensor
// self: grad * self.cosh()
std::vector<Tensor> ExecuteUnaryBackwardSinh::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = ttnn::multiply(
        ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor grad_a = where(
        ttnn::gt(input, 88.5f, std::nullopt, output_mem_config),
        t_inf,
        where(
            ttnn::lt(input, -88.5f, std::nullopt, output_mem_config),
            t_inf,
            ttnn::multiply(grad, ttnn::cosh(input, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        output_mem_config);
    t_inf.deallocate();
    grad_a = where(
        ttnn::ge(grad_a, 3.4e+38, std::nullopt, output_mem_config),
        std::numeric_limits<float>::infinity(),
        where(
            ttnn::le(grad_a, -3.4e+38, std::nullopt, output_mem_config),
            -std::numeric_limits<float>::infinity(),
            grad_a,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// bw(log10(in)) = grad/(in * 2.30258509299404568402)
std::vector<Tensor> ExecuteUnaryBackwardLog10::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor grad_a = ttnn::multiply(
        grad,
        ttnn::reciprocal(ttnn::multiply(input, M_LN10, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        std::nanf(" "),
        where(ttnn::eqz(input, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// bw(log1p(in)) = grad/(in + 1)
// for -1 = inf
std::vector<Tensor> ExecuteUnaryBackwardLog1p::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor t_inp1 = ttnn::add(input, 1.0f, std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(t_inp1, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(ttnn::eq(input, -1.0f, std::nullopt, output_mem_config), t_inf, grad_a, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(t_inp1, output_mem_config), eqz(grad, output_mem_config)),
        std::nanf(" "),
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardErfc::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        ttnn::multiply(
            ttnn::exp(ttnn::neg(ttnn::square(input, output_mem_config), output_mem_config), false, output_mem_config),
            grad,
            std::nullopt,
            output_mem_config),
        -M_2_SQRTPI,
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardCeil::invoke(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = ttnn::zeros_like(grad);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}

// softsign
// result = grad_data / torch.square(1 + torch.abs(input))
std::vector<Tensor> ExecuteUnaryBackwardSoftsign::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::ABS},
        EltwiseUnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
        EltwiseUnaryWithParam{UnaryOpType::SQUARE},
        EltwiseUnaryWithParam{UnaryOpType::RECIP}};
    grad_tensor.emplace_back(
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config));
    return grad_tensor;
}

// name: cosh(Tensor self) -> Tensor
// self: grad * self.sinh()
std::vector<Tensor> ExecuteUnaryBackwardCosh::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = ttnn::multiply(
        ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor t_neg_inf = ttnn::multiply(
        ttnn::sign(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor grad_a = where(
        ttnn::gt(input, 88.50f, std::nullopt, output_mem_config),
        t_inf,
        where(
            ttnn::lt(input, -88.50f, std::nullopt, output_mem_config),
            t_neg_inf,
            ttnn::multiply(grad, ttnn::sinh(input, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        output_mem_config);
    t_neg_inf.deallocate();
    t_inf.deallocate();
    grad_a = where(
        ttnn::ge(grad_a, 3.4e+38, std::nullopt, output_mem_config),
        std::numeric_limits<float>::infinity(),
        where(
            ttnn::le(grad_a, -3.4e+38, std::nullopt, output_mem_config),
            -std::numeric_limits<float>::infinity(),
            grad_a,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// Torch reference
// # if eps is not None:
// #         lo = eps
// #         hi = 1.0 - lo
// #         return torch.where(
// #             torch.ttnn::logical_and(self >= lo, self <= hi),
// #             grad_output / (self * (1.0 - self)),
// #             0.0,
// #         )
// #     else:
// #         return torch.where(
// #             torch.ttnn::logical_and(self >= 0.0, self <= 1.0),
// #             grad_output / (self * (1.0 - self)),
// #             self.new_full((), float("nan")),
// #         )
std::vector<Tensor> ExecuteUnaryBackwardLogiteps::invoke(
    const Tensor& grad, const Tensor& input, float eps, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float low, high;
    low = eps;
    high = 1.0 - low;
    Tensor grad_result = ttnn::multiply(
        grad,
        ttnn::reciprocal(ttnn::multiply(
            input, ttnn::rsub(input, 1.0f, std::nullopt, output_mem_config), std::nullopt, output_mem_config)),
        std::nullopt,
        output_mem_config);
    Tensor t_eps = ttnn::full_like(input, eps, input.dtype(), input.layout(), std::nullopt, output_mem_config);
    Tensor ltl_gth = ttnn::logical_or(
        ttnn::lt(input, low, std::nullopt, output_mem_config),
        ttnn::gt(input, high, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = where(
        ttnn::eq(ltl_gth, 1.0f, std::nullopt, output_mem_config),
        where(ttnn::ltz(t_eps, output_mem_config), std::nanf(" "), 0.f, output_mem_config),
        where(
            ttnn::logical_or(
                ttnn::eq(input, 0.0, std::nullopt, output_mem_config),
                ttnn::eq(input, 1.0, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            ttnn::multiply(
                ttnn::sign(grad, output_mem_config),
                std::numeric_limits<float>::infinity(),
                std::nullopt,
                output_mem_config),
            grad_result,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// bw(log2(in)) = grad/(in * 0.69314718055994530942)
std::vector<Tensor> ExecuteUnaryBackwardLog2::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor grad_a = ttnn::multiply(
        grad,
        ttnn::reciprocal(ttnn::multiply(input, M_LN2, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        std::nanf(" "),
        where(ttnn::eqz(input, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardSign::invoke(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = ttnn::zeros_like(grad, grad.dtype(), grad.layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardDivNoNan::invoke(
    const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor val = ttnn::full_like(input, scalar, input.dtype(), input.layout(), std::nullopt, output_mem_config);
    Tensor result = where(
        ttnn::eq(val, 0, std::nullopt, output_mem_config),
        0.f,
        ttnn::multiply(grad, 1 / scalar, std::nullopt, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// #  bw (exp2) = grad * exp2(input) * M_LN2
// # M_LN2 = 0.693147180559945309417
std::vector<Tensor> ExecuteUnaryBackwardExp2::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor exp_result = ttnn::exp2(input, output_mem_config);
    exp_result = ttnn::multiply(exp_result, M_LN2, std::nullopt, output_mem_config);
    Tensor result = ttnn::multiply(grad, exp_result, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// bw(expm1) = grad * expm1(input) + 1
std::vector<Tensor> ExecuteUnaryBackwardExpm1::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor eresult = ttnn::expm1(input, output_mem_config);
    Tensor rp1 = ttnn::add(eresult, 1.0f, std::nullopt, output_mem_config);
    Tensor result = ttnn::multiply(grad, rp1, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardRecip::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");
    grad_tensor.emplace_back(where(
        ttnn::eqz(input, output_mem_config),
        where(
            ttnn::eqz(grad, output_mem_config),
            t_nan,
            ttnn::multiply(
                ttnn::neg(ttnn::sign(grad, output_mem_config), output_mem_config),
                t_inf,
                std::nullopt,
                output_mem_config),
            output_mem_config),
        ttnn::multiply(
            ttnn::neg(grad, output_mem_config),
            ttnn::reciprocal(ttnn::square(input, output_mem_config), output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config));
    return grad_tensor;
}

std::vector<ComplexTensor> ExecuteUnaryBackwardRecip::invoke(
    const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor condition_nan = ttnn::logical_and(
        ttnn::eqz(input.real(), output_mem_config),
        ttnn::eqz(input.imag(), output_mem_config),
        std::nullopt,
        output_mem_config);
    ComplexTensor neg_grad =
        ComplexTensor({ttnn::neg(grad.real(), output_mem_config), ttnn::neg(grad.imag(), output_mem_config)});
    ComplexTensor inp_recip = ttnn::reciprocal(input, output_mem_config);
    ComplexTensor grad_inp = ttnn::operations::complex_binary::_mul(
        neg_grad,
        ttnn::conj(ttnn::operations::complex_binary::_mul(inp_recip, inp_recip, output_mem_config), output_mem_config),
        output_mem_config);
    neg_grad.deallocate();
    inp_recip.deallocate();
    Tensor grad_inp_r = where(condition_nan, std::nanf(""), grad_inp.real(), output_mem_config);
    Tensor grad_inp_i = where(condition_nan, std::nanf(""), grad_inp.imag(), output_mem_config);
    condition_nan.deallocate();
    grad_inp = ComplexTensor({grad_inp_r, grad_inp_i});
    grad_inp_r.deallocate();
    grad_inp_i.deallocate();
    grad_tensor.emplace_back(grad_inp);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardAbs::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(grad, ttnn::sign(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<ComplexTensor> ExecuteUnaryBackwardAbs::invoke(
    const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor result = ttnn::abs(input, output_mem_config);
    Tensor grad_inp_r = where(
        ttnn::eqz(result, output_mem_config),
        0.f,
        ttnn::multiply(
            grad,
            ttnn::multiply(input.real(), ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    Tensor grad_inp_i = where(
        ttnn::eqz(result, output_mem_config),
        0.f,
        ttnn::multiply(
            grad,
            ttnn::multiply(input.imag(), ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    ComplexTensor grad_inp = ComplexTensor({grad_inp_r, grad_inp_i});
    result.deallocate();
    grad_inp_r.deallocate();
    grad_inp_i.deallocate();
    grad_tensor.emplace_back(grad_inp);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardDigamma::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");
    Tensor grad_a = ttnn::multiply(grad, ttnn::polygamma(input, 1, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_nan,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
        -t_inf,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardPolygamma::invoke(
    const Tensor& grad, const Tensor& input, int n, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    float t_nan = std::nanf("");
    float pos_neg = 1.0f;
    if (n == 2 || n == 4 || n == 6 || n == 8 || n == 10) {
        pos_neg = -1.0f;
    }
    Tensor grad_a =
        ttnn::multiply(grad, ttnn::polygamma(input, (n + 1), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::le(input, 0.0, std::nullopt, output_mem_config),
            ttnn::eqz(grad, output_mem_config),
            std::nullopt,
            output_mem_config),
        t_nan,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
        (-std::numeric_limits<float>::infinity() * pos_neg),
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
        (std::numeric_limits<float>::infinity() * pos_neg),
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// erfinv
// self: 0.5 * sqrt(M_PI) * exp(self.erfinv().pow(2)) * grad
// for input -1 and 1: grad.sign() * inf, for input > 1 or < -1 : nan
std::vector<Tensor> ExecuteUnaryBackwardErfinv::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float m_sqrtpi = 1.77245385090551602792981f;
    Tensor result = ttnn::multiply(
        ttnn::multiply(
            ttnn::multiply(
                ttnn::exp(
                    ttnn::square(ttnn::erfinv(input, output_mem_config), output_mem_config), false, output_mem_config),
                grad,
                std::nullopt,
                output_mem_config),
            m_sqrtpi,
            std::nullopt,
            output_mem_config),
        0.5,
        std::nullopt,
        output_mem_config);
    Tensor t_inf = ttnn::multiply(
        ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    result = ttnn::where(
        ttnn::logical_or(
            ttnn::lt(input, -1.0f, std::nullopt, output_mem_config),
            ttnn::gt(input, 1.0f, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        std::nanf(" "),
        result,
        output_mem_config);
    result = ttnn::where(
        ttnn::eq(input, -1.0f, std::nullopt, output_mem_config),
        t_inf,
        ttnn::where(ttnn::eq(input, 1.0f, std::nullopt, output_mem_config), t_inf, result, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardErf::invoke(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        ttnn::multiply(
            ttnn::exp(ttnn::neg(ttnn::square(input, output_mem_config), output_mem_config), false, output_mem_config),
            grad,
            std::nullopt,
            output_mem_config),
        M_2_SQRTPI,
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> ExecuteUnaryBackwardDeg2rad::invoke(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float M_PI_180 = M_PI / 180;
    Tensor grad_result = ttnn::multiply(grad, M_PI_180, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<std::optional<ttnn::Tensor>> ExecuteUnaryBackwardGelu::invoke(
    const Tensor& grad,
    const Tensor& input,
    const std::string& approximate,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result;
    if (!input_grad.has_value()) {
        input_grad = ttnn::empty_like(grad);
    }

    auto output_memory_config =
        input_grad.has_value() ? input_grad->memory_config() : output_mem_config.value_or(input.memory_config());
    TT_FATAL((approximate == "none" || approximate == "tanh"), "Incorrect approximate mode (expected 'None', 'tanh')");

    if (approximate == "tanh") {
        float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
        float kKappa = 0.044715;
        Tensor x_sq = ttnn::multiply(input, input, std::nullopt, output_memory_config);
        Tensor x_cube = ttnn::multiply(x_sq, input, std::nullopt, output_memory_config);
        Tensor inner = ttnn::multiply(
            ttnn::add(input, ttnn::multiply(x_cube, kKappa, std::nullopt, output_memory_config)),
            kBeta,
            std::nullopt,
            output_mem_config);
        Tensor tanh_inner = ttnn::tanh(inner, output_memory_config);

        Tensor left = ttnn::multiply(input, 0.5, std::nullopt, output_memory_config);
        Tensor right = ttnn::add(tanh_inner, 1, std::nullopt, output_memory_config);

        Tensor left_derivative = ttnn::multiply(right, 0.5, std::nullopt, output_memory_config);

        Tensor tanh_derivative = ttnn::neg(
            ttnn::subtract(
                ttnn::multiply(tanh_inner, tanh_inner, std::nullopt, output_memory_config),
                1,
                std::nullopt,
                output_mem_config),
            output_memory_config);
        Tensor inner_derivative = ttnn::multiply(
            (ttnn::add(
                ttnn::multiply(
                    ttnn::multiply(x_sq, kKappa, std::nullopt, output_memory_config),
                    3,
                    std::nullopt,
                    output_memory_config),
                1,
                std::nullopt,
                output_mem_config)),
            kBeta);
        Tensor right_derivative = ttnn::multiply(
            ttnn::multiply(tanh_derivative, left, std::nullopt, output_memory_config),
            inner_derivative,
            std::nullopt,
            output_memory_config);

        ttnn::multiply(
            grad, (ttnn::add(left_derivative, right_derivative)), std::nullopt, output_memory_config, input_grad);
        result.push_back(input_grad);
    } else {
        float kAlpha = M_SQRT1_2;
        float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
        Tensor cdf = ttnn::multiply(
            (ttnn::add(
                ttnn::erf(ttnn::multiply(input, kAlpha, std::nullopt, output_memory_config)),
                1,
                std::nullopt,
                output_memory_config)),
            0.5);
        Tensor pdf = ttnn::multiply(
            ttnn::exp(ttnn::multiply(ttnn::multiply(input, input), -0.5), false, output_memory_config),
            kBeta,
            std::nullopt,
            output_memory_config);
        ttnn::multiply(
            grad, ttnn::add(cdf, ttnn::multiply(input, pdf)), std::nullopt, output_memory_config, input_grad);
        result.push_back(input_grad);
    }

    return result;
}

std::vector<Tensor> ExecuteUnaryBackwardRepeat::invoke(
    const Tensor& grad,
    const Tensor& input,
    const ttnn::Shape& shape,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(
        input.memory_config());  // TODO: Remove after ternary forward ops migration is completed

    auto shape_wh = input.padded_shape();
    TT_FATAL(shape_wh[0] == 1, "Input shape[0] must be 1 but got {}", shape_wh[0]);
    auto* ttnn_device = input.device();
    // input.padded_shape()[0]
    // If repeat shape has 0's, it returns zeros of given input
    if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0 || shape[3] == 0) {
        Tensor zero_tensor = ttnn::zeros_like(input, input.dtype(), input.layout(), std::nullopt, output_memory_config);
        grad_tensor.emplace_back(zero_tensor);
        return grad_tensor;
    }
    if (shape[0] > 1) {
        ttnn::SmallVector<int64_t> dim = {0};
        TT_FATAL(shape[1] == 1 && shape[2] == 1 && shape[3] == 1, "repeat[1], [2], [3] should be 1");
        std::array<std::uint32_t, 4> intended_shape_array = {1, shape_wh[1], shape_wh[2], shape_wh[3]};
        const auto required = ttnn::Shape(intended_shape_array);
        Tensor result = ttnn::moreh_sum(
            grad,
            dim,
            true,
            ttnn::zeros(required, input.dtype(), input.layout(), *ttnn_device, output_memory_config),
            output_memory_config,
            std::nullopt);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    }
    if (shape[1] > 1) {
        ttnn::SmallVector<int64_t> dim = {1};
        TT_FATAL(shape[0] == 1 && shape[2] == 1 && shape[3] == 1, "repeat[0], [2], [3] should be 1");
        std::array<std::uint32_t, 4> intended_shape_array = {shape_wh[0], 1, shape_wh[2], shape_wh[3]};
        const auto required = ttnn::Shape(intended_shape_array);
        Tensor result = ttnn::moreh_sum(
            grad,
            dim,
            true,
            ttnn::zeros(required, input.dtype(), input.layout(), *ttnn_device, output_memory_config),
            output_memory_config,
            std::nullopt);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    }
    return grad_tensor;
}

// Autoformat support
Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& /*output_mem_config*/) {
    auto formatted_input_tensor = temp;
    if (formatted_input_tensor.layout() == Layout::ROW_MAJOR) {
        auto a_pad_shape = ttnn::operations::data_movement::pad_to_tile_shape(temp.padded_shape());
        auto need_format = temp.layout() != Layout::TILE || temp.padded_shape() != a_pad_shape;
        if (need_format) {
            formatted_input_tensor =
                ttnn::tilize_with_val_padding(temp, a_pad_shape, PadValue(1.0f), temp.memory_config());
        }
    }
    return formatted_input_tensor;
}

// Prod
// along a single dimension --> result: grad_data * (y / input )
std::vector<Tensor> ExecuteUnaryBackwardProd::invoke(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<int64_t> dim,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(
        input.memory_config());  // TODO: Remove after ternary forward ops migration is completed

    const bool all_dimensions = !dim.has_value();
    const bool keepdim = !all_dimensions;
    Tensor prod_result = ttnn::prod(input, dim, keepdim, output_memory_config);

    if (prod_result.layout() == Layout::ROW_MAJOR && prod_result.storage_type() == StorageType::DEVICE) {
        prod_result = ttnn::operations::unary_backward::change_layout_to_tile(prod_result, output_memory_config);
    }

    if (all_dimensions) {
        Tensor temp = ttnn::multiply(
            prod_result, grad, std::nullopt, output_memory_config);  // result is stored in the first position
        Tensor fill_tensor = ttnn::fill_first_val_into_tensor<::bfloat16>(
            temp, temp.dtype(), temp.layout(), temp.device(), output_memory_config);
        Tensor all_dimension_result = ttnn::multiply(
            ttnn::reciprocal(input, output_memory_config), fill_tensor, std::nullopt, output_memory_config);
        grad_tensor.emplace_back(all_dimension_result);
        return grad_tensor;
    }

    // all_dimensions = False
    Tensor updated_grad = prod_result;
    auto step = ttnn::SmallVector<uint32_t>({1, 1, 1, 1});
    if (prod_result.logical_shape() != grad.padded_shape()) {
        if (*dim == 3 || *dim == -1) {
            ttnn::SmallVector<int64_t> after_permute_dims = {0, 3, 1, 2};
            Tensor required = ttnn::permute(grad, after_permute_dims, output_memory_config);
            ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
            ttnn::SmallVector<uint32_t> end_index = {
                grad.padded_shape()[0], 1, grad.padded_shape()[1], grad.padded_shape()[2]};
            Tensor new_slice_tensor = ttnn::slice(required, start_index, end_index, step, std::nullopt);
            after_permute_dims = {0, 2, 3, 1};
            updated_grad = ttnn::permute(new_slice_tensor, after_permute_dims, output_memory_config);
            if (updated_grad.storage_type() != StorageType::DEVICE) {
                Tensor pad_updated_grad = updated_grad.pad_to_tile(1.0f);
                pad_updated_grad = pad_updated_grad.to_layout(Layout::TILE);
                updated_grad = pad_updated_grad.to_device(input.device());
            }
        } else if (*dim == 2 || *dim == -2) {
            ttnn::SmallVector<int64_t> after_permute_dims = {0, 2, 1, 3};
            Tensor required = ttnn::permute(grad, after_permute_dims, output_memory_config);
            ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
            ttnn::SmallVector<uint32_t> end_index = {
                grad.padded_shape()[0], 1, grad.padded_shape()[1], grad.padded_shape()[3]};
            Tensor new_slice_tensor = ttnn::slice(required, start_index, end_index, step, std::nullopt);
            updated_grad = ttnn::permute(new_slice_tensor, after_permute_dims, output_memory_config);
            if (updated_grad.layout() == Layout::ROW_MAJOR) {
                updated_grad =
                    ttnn::operations::unary_backward::change_layout_to_tile(updated_grad, output_memory_config);
            }
        }
    }
    Tensor reciprocal_input = ttnn::reciprocal(input, output_memory_config);
    Tensor temp = ttnn::multiply(
        prod_result,
        (*dim == 1 || *dim == 0 || *dim == -4 || *dim == -3) ? grad : updated_grad,
        std::nullopt,
        output_memory_config);
    if (temp.layout() == Layout::ROW_MAJOR) {
        temp = ttnn::operations::unary_backward::change_layout_to_tile(temp, output_memory_config);
    }
    if (*dim == 3 || *dim == -1) {
        Tensor grad_result =
            ttnn::bcast(reciprocal_input, temp, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::W, output_memory_config);
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    }
    if (*dim == 2 || *dim == -2) {
        Tensor grad_result =
            ttnn::bcast(reciprocal_input, temp, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::H, output_memory_config);
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    }
    if (*dim == 1 || *dim == -3) {
        Tensor tensor_1_temp = reciprocal_input;
        if (reciprocal_input.padded_shape()[1] % 32 != 0) {
            ttnn::SmallVector<std::array<uint32_t, 2>> padding = {
                {0, 0}, {0, 32 - (reciprocal_input.padded_shape()[1] % 32)}, {0, 0}, {0, 0}};
            tensor_1_temp = ttnn::pad(reciprocal_input, padding, 0, true, std::nullopt);
        }
        ttnn::SmallVector<int64_t> after_permute_dims = {0, 2, 3, 1};
        Tensor tensor_1 = ttnn::permute(tensor_1_temp, after_permute_dims, output_memory_config);
        Tensor tensor_2 = ttnn::permute(temp, after_permute_dims, output_memory_config);

        // put the tensor back on device because permute throws it off device
        // See: Remove auto format within permute_op.cpp #9404
        auto padded_shape = ttnn::operations::data_movement::pad_to_tile_shape(tensor_1.padded_shape());
        // tensor_2 is always TILE layout (from permute of TILE temp)
        // Only need to convert if tensor_1 is ROW_MAJOR
        tensor_2 = tensor_2.to_device(tensor_1.device());
        if (tensor_1.layout() == Layout::ROW_MAJOR) {
            // Need to untilize tensor_2 to match tensor_1's ROW_MAJOR layout
            bool pad_needed = tensor_2.padded_shape() != padded_shape;
            tensor_2 = ttnn::untilize(tensor_2, tensor_1.memory_config());
            if (pad_needed) {
                tensor_2 = ttnn::pad(
                    tensor_2,
                    padded_shape.to_array_4D(),
                    tt::tt_metal::Array4D({0, 0, 0, 0}),
                    0.0f,
                    false,
                    tensor_1.memory_config());
            }
        }
        // If tensor_1 is TILE, tensor_2 is already correct (both TILE, shapes match by assumption)

        after_permute_dims = {0, 3, 1, 2};
        Tensor result = permute(
            ttnn::bcast(tensor_1, tensor_2, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::W, output_memory_config),
            after_permute_dims,
            output_memory_config);
        Tensor grad_result = result;
        if (reciprocal_input.padded_shape()[1] % 32 != 0) {
            ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
            ttnn::SmallVector<uint32_t> end_index = {
                input.padded_shape()[0], input.padded_shape()[1], input.padded_shape()[2], input.padded_shape()[3]};
            auto step = ttnn::SmallVector<uint32_t>({1, 1, 1, 1});
            grad_result = ttnn::slice(result, start_index, end_index, step, std::nullopt);
        }
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    }
    // dim 0
    Tensor tensor_1_temp = reciprocal_input;
    if (reciprocal_input.padded_shape()[0] % 32 != 0) {
        ttnn::SmallVector<std::array<uint32_t, 2>> padding = {
            {0, (32 - (reciprocal_input.padded_shape()[0] % 32))}, {0, 0}, {0, 0}, {0, 0}};
        tensor_1_temp = ttnn::pad(reciprocal_input, padding, 0, false, std::nullopt);
    }
    ttnn::SmallVector<int64_t> after_permute_dims = {3, 1, 2, 0};
    Tensor tensor_1 = ttnn::permute(tensor_1_temp, after_permute_dims, output_memory_config);
    Tensor tensor_2 = ttnn::permute(temp, after_permute_dims, output_memory_config);

    // put the tensor back on device because permute throws it off device
    // See: Remove auto format within permute_op.cpp #9404
    auto padded_shape = ttnn::operations::data_movement::pad_to_tile_shape(tensor_2.padded_shape());
    // tensor_2 is always TILE layout (from permute of TILE temp)
    // Only need to convert if tensor_1 is ROW_MAJOR
    tensor_2 = tensor_2.to_device(tensor_1.device());
    if (tensor_1.layout() == Layout::ROW_MAJOR) {
        // Need to untilize tensor_2 to match tensor_1's ROW_MAJOR layout
        bool pad_needed = tensor_2.padded_shape() != padded_shape;
        tensor_2 = ttnn::untilize(tensor_2, tensor_1.memory_config());
        if (pad_needed) {
            tensor_2 = ttnn::pad(
                tensor_2,
                padded_shape.to_array_4D(),
                tt::tt_metal::Array4D({0, 0, 0, 0}),
                0.0f,
                false,
                tensor_1.memory_config());
        }
    }
    // If tensor_1 is TILE, tensor_2 is already correct (both TILE, shapes match by assumption)

    Tensor result = ttnn::permute(
        ttnn::bcast(tensor_1, tensor_2, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::W, output_memory_config),
        after_permute_dims,
        output_memory_config);
    Tensor grad_result = result;
    if (reciprocal_input.padded_shape()[0] % 32 != 0) {
        ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end_index = {
            input.padded_shape()[0], input.padded_shape()[1], input.padded_shape()[2], input.padded_shape()[3]};
        grad_result = ttnn::slice(result, start_index, end_index, step, std::nullopt);
    }
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

}  // namespace ttnn::operations::unary_backward
