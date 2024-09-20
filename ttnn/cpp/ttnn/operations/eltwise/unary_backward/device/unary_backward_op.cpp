// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/unary_backward/device/unary_backward_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"
#include "ttnn/operations/eltwise/complex_binary/device/complex_binary_op.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::unary_backward {

std::vector<Tensor> ExecuteUnaryBackwardClamp::invoke(
    const Tensor& grad, const Tensor& input, std::optional<float> min, std::optional<float> max, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config()); //TODO: Remove after ternary forward ops migration is completed
    TT_FATAL((max.has_value() || min.has_value()) && "Only one of 'min' or 'max' can be None. Please provide atleast one value", "Error");
    if (!max.has_value()) {
        Tensor minT = ttnn::ge(input, min.value(), std::nullopt, output_mem_config);
        Tensor result = ttnn::multiply(grad, minT, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(result);
    return grad_tensor;
    }else if(!min.has_value()) {
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

// Hardtanh
// result: torch.where((input <= min) | (input >= max), 0.0, grad)
std::vector<Tensor> ExecuteUnaryBackwardHardtanh::invoke(
    const Tensor& grad, const Tensor& input, float min, float max, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::where(
        ttnn::le(input, ttnn::full_like(input, min), std::nullopt, output_mem_config),
        0.0,
        ttnn::where(ttnn::ge(input, ttnn::full_like(input, max), std::nullopt, output_mem_config), 0.0, grad),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// threshold
// if input <= threshold = 0 else grad
std::vector<Tensor> ExecuteUnaryBackwardThreshold::invoke(
    const Tensor& grad, const Tensor& input, float threshold, float value, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::where(
        ttnn::gtz(ttnn::add(input, -threshold, std::nullopt, output_mem_config), output_mem_config),
        grad,
        ttnn::zeros_like(grad, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// Softplus
std::vector<Tensor> ExecuteUnaryBackwardSoftplus::invoke(
    const Tensor& grad, const Tensor& input, float beta, float threshold, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor mul_input_beta = ttnn::multiply(input, beta, std::nullopt, output_mem_config);
    Tensor exp_beta_self = ttnn::exp(mul_input_beta, false, output_mem_config);
    Tensor sub_result = ttnn::add(mul_input_beta, -threshold, std::nullopt, output_mem_config);
    Tensor temp =
        ttnn::multiply(ttnn::multiply(grad, exp_beta_self, std::nullopt, output_mem_config),
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

std::vector<Tensor> _rdiv_bw(
    const Tensor& grad, const Tensor& input, float scalar, string round_mode, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    TT_FATAL((round_mode == "None" || round_mode == "trunc" || round_mode == "floor"), "Incorrect rounding mode (expected 'None', 'trunc', or 'floor')");
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    if (round_mode == "None") {
        Tensor result = ttnn::where(
            ttnn::nez(input),
            ttnn::multiply(ttnn::neg(grad, output_mem_config),
                (ttnn::multiply(ttnn::reciprocal(ttnn::square(input, output_mem_config)), scalar, std::nullopt, output_mem_config)),
                std::nullopt,
                output_mem_config),
            t_nan,
            output_mem_config);
        if (scalar > 0) {
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
                t_inf,
                result,
                output_mem_config);
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
                -t_inf,
                result,
                output_mem_config);
        } else if (scalar < 0) {
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
                -t_inf,
                result,
                output_mem_config);
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
                t_inf,
                result,
                output_mem_config);
        }
        grad_tensor.emplace_back(result);
    } else {
        Tensor result = ttnn::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(result);
    }
    return grad_tensor;
}

// unary_pow:
// grad_input = grad * exponent * torch.pow(input, exponent - 1)
std::vector<std::optional<Tensor>> ExecuteUnaryBackwardPow::invoke(uint8_t queue_id, const Tensor& grad, const Tensor& input, float exponent, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;
    input_grad = input_grad.value_or(ttnn::empty_like(input));
    const float ZERO_THRESHOLD = std::numeric_limits<float>::epsilon() * 10.0f;
    TT_FATAL(exponent >= 0.0, "negative exponents are not supported; use recip(pow(input,abs(exponent)))");
    if (std::abs(exponent) < ZERO_THRESHOLD) {
        input_grad = ttnn::full_like(input, 0.0f);
        grad_tensor.emplace_back(input_grad);
        return grad_tensor;
    }

    Tensor power_input = ttnn::power(queue_id,input, fabs(exponent - 1.0f), output_mem_config);
    if (exponent < 1.0f) {
        power_input = ttnn::reciprocal(queue_id, power_input, output_mem_config);
    }

    Tensor result = ttnn::multiply(queue_id, power_input, exponent, std::nullopt, output_mem_config);
    power_input.deallocate();
    Tensor final_result = ttnn::multiply(queue_id, result, grad, std::nullopt, output_mem_config);
    result.deallocate();
    Tensor temp = where(queue_id, ttnn::le(queue_id, final_result, -3.4e+38, std::nullopt, output_mem_config), -std::numeric_limits<float>::infinity(), final_result, output_mem_config);
        where(queue_id, ttnn::ge(queue_id, final_result, 3.4e+38, std::nullopt, output_mem_config), std::numeric_limits<float>::infinity(), temp, output_mem_config, input_grad);
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardPow::invoke(const Tensor& grad, const Tensor& input, float exponent, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    return ExecuteUnaryBackwardPow::invoke(DefaultQueueId, grad, input, exponent, output_mem_config, input_grad);
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardExp::invoke(uint8_t queue_id, const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;

    input_grad = input_grad.value_or(ttnn::empty_like(input));
    float t_inf = std::numeric_limits<float>::infinity();
    Tensor exp_result = ttnn::exp(queue_id, input, false, output_mem_config);
    Tensor result = ttnn::multiply(queue_id, grad, exp_result, std::nullopt, output_mem_config);
    result = where(queue_id, ttnn::ge(queue_id, result, 1e+38, std::nullopt, output_mem_config), t_inf, result, output_mem_config);
    result = where(queue_id, ttnn::ge(queue_id, result, -1e+38, std::nullopt,  output_mem_config), -t_inf, result, output_mem_config);
    where(queue_id,
    ttnn::logical_and(
        ttnn::ge(queue_id, ttnn::abs(queue_id, exp_result, output_mem_config), 1e+38, std::nullopt, output_mem_config),
        ttnn::ltz(queue_id, grad, output_mem_config), std::nullopt, output_mem_config), -t_inf, result, output_mem_config, input_grad);

    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardExp::invoke(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    return ExecuteUnaryBackwardExp::invoke(DefaultQueueId, grad, input, output_mem_config, input_grad);
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardTanh::invoke(uint8_t queue_id, const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;

    input_grad = input_grad.value_or(ttnn::empty_like(input));
    Tensor tanh_res = ttnn::tanh(queue_id, input, output_mem_config);
    tanh_res = ttnn::square(queue_id, tanh_res, output_mem_config);
    tanh_res = ttnn::rsub(queue_id, tanh_res, 1.0f, output_mem_config);
    ttnn::multiply(queue_id, grad, tanh_res, std::nullopt, output_mem_config, input_grad);
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardTanh::invoke(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    return ExecuteUnaryBackwardTanh::invoke(DefaultQueueId, grad, input, output_mem_config, input_grad);
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardSqrt::invoke(uint8_t queue_id, const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;

    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();

    input_grad = input_grad.value_or(ttnn::empty_like(input));
    ttnn::sqrt(queue_id, input, output_mem_config, input_grad);
    ttnn::multiply(queue_id, grad, ttnn::reciprocal(queue_id, ttnn::multiply(queue_id, input_grad.value(), 2.0, std::nullopt, output_mem_config), output_mem_config),std::nullopt,output_mem_config, input_grad);
    where(queue_id, ttnn::lez(queue_id, input, output_mem_config), t_nan, input_grad.value(), output_mem_config, input_grad);
    where(queue_id, ttnn::logical_and(queue_id, ttnn::eqz(queue_id, input, output_mem_config), ttnn::ltz(queue_id, grad, output_mem_config), std::nullopt, output_mem_config), -t_inf, input_grad.value(),output_mem_config,input_grad);
    where(queue_id, ttnn::logical_and(queue_id, ttnn::eqz(queue_id, input, output_mem_config), ttnn::gtz(queue_id, grad, output_mem_config), std::nullopt, output_mem_config), t_inf, input_grad.value(),output_mem_config,input_grad);
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardSqrt::invoke(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    return ExecuteUnaryBackwardSqrt::invoke(DefaultQueueId, grad, input, output_mem_config, input_grad);
}

std::vector<Tensor> _multigammaln_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor digamma_result = ttnn::multiply(grad, ttnn::digamma(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor digamma_result_2 = ttnn::multiply(
        grad, ttnn::digamma(ttnn::add(input, -0.5, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);

    Tensor grad_result = ttnn::add(digamma_result, digamma_result_2, std::nullopt, output_mem_config);

    digamma_result = ttnn::multiply(
        grad, ttnn::digamma(ttnn::add(input, -1.0, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_result = ttnn::add(grad_result, digamma_result, std::nullopt, output_mem_config);

    digamma_result = ttnn::multiply(
        grad, ttnn::digamma(ttnn::add(input, -1.5, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_result = ttnn::add(grad_result, digamma_result, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


std::vector<Tensor> _unary_comp_bw(const Tensor& grad, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = ttnn::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}

std::vector<Tensor> _eq_bw(
    const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config) {
    return _unary_comp_bw(grad, output_mem_config);
}

std::vector<Tensor> _gt_bw(
    const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config) {
    return _unary_comp_bw(grad, output_mem_config);
}

std::vector<Tensor> _ge_bw(
    const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config) {
    return _unary_comp_bw(grad, output_mem_config);
}

std::vector<Tensor> _lgamma_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    auto output_memory_config = output_mem_config.value_or(input.memory_config()); //TODO: Remove after ternary forward ops migration is completed
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(grad, ttnn::digamma(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _sub_bw(const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _frac_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _trunc_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// return: grad_output * (max_deriv - sign * (z / (1 + z)))
// z = exp(-abs(input))
std::vector<Tensor> _log_sigmoid_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor max_deriv = ttnn::where(ttnn::ltz(input, output_mem_config), 1, 0, output_mem_config);
    Tensor in_sign = ttnn::where(ttnn::ltz(input, output_mem_config), 1, -1, output_mem_config);
    Tensor in_abs = ttnn::abs(input, output_mem_config);
    Tensor z = ttnn::exp(ttnn::neg(in_abs, output_mem_config), false, output_mem_config);

    Tensor mul_z = ttnn::multiply(z, ttnn::reciprocal((ttnn::add(z, 1.0f, std::nullopt, output_mem_config)), output_mem_config), std::nullopt, output_mem_config);

    Tensor mul_sign = ttnn::multiply(in_sign, mul_z, std::nullopt, output_mem_config);
    Tensor sub_max = ttnn::subtract(max_deriv, mul_sign, std::nullopt, output_mem_config);

    Tensor grad_result = ttnn::multiply(grad, sub_max, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _fill_zero_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _i0_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_inf = std::numeric_limits<float>::infinity();
    Tensor value = ttnn::multiply(
        ttnn::multiply(ttnn::i0(input, output_mem_config), ttnn::reciprocal(input, output_mem_config), std::nullopt, output_mem_config),
        0.5,
        std::nullopt,
        output_mem_config);
    Tensor result = ttnn::where(
        ttnn::ltz(input, output_mem_config),
        ttnn::multiply(grad,
            ttnn::subtract(ttnn::neg(ttnn::i0(input, output_mem_config), output_mem_config), value, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        ttnn::multiply(grad,
            ttnn::subtract(ttnn::i0(input, output_mem_config), value, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    result = ttnn::where(
        ttnn::ge(ttnn::abs(ttnn::i0(input, output_mem_config), output_mem_config), 3.4e+38, std::nullopt, output_mem_config),
        t_inf,
        result,
        output_mem_config);
    result =
        ttnn::where(ttnn::ge(ttnn::abs(result, output_mem_config), 3.4e+38, std::nullopt, output_mem_config), t_inf, result, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _tan_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tan_result = ttnn::tan(input, output_mem_config);
    Tensor result =
        ttnn::multiply(grad, ttnn::add(ttnn::square(tan_result, output_mem_config), 1.0f, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// grad(sigmoid) = grad*(1 - sigmoid(x))*sigmoid(x)
std::vector<Tensor> _sigmoid_bw(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    std::vector<Tensor> grad_tensor;
    Tensor sig_result = ttnn::sigmoid(input, output_mem_config);
    Tensor rsub_term = ttnn::rsub(sig_result, 1.0f, output_mem_config);
    Tensor prod_term_1 = ttnn::multiply(sig_result, rsub_term, std::nullopt, output_mem_config);
    Tensor prod_term_2 = ttnn::multiply(prod_term_1, grad, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(prod_term_2);
    return grad_tensor;
}

std::vector<std::optional<ttnn::Tensor>> ExecuteUnaryBackwardRsqrt::invoke(uint8_t queue_id, const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result;
    if(!input_grad.has_value()){
        input_grad = ttnn::empty_like(grad);
    }
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");

    ttnn::rsqrt(queue_id, input, true, output_mem_config, input_grad);
    ttnn::power(queue_id, input_grad.value(), 3, output_mem_config, input_grad);
    ttnn::multiply(queue_id, ttnn::multiply(queue_id, grad, input_grad.value(), std::nullopt, output_mem_config), -0.5, std::nullopt, output_mem_config, input_grad);
    where(queue_id, ttnn::eqz(queue_id, input, output_mem_config), t_inf, input_grad.value(), output_mem_config, input_grad);
    where(queue_id, ttnn::ltz(queue_id, input, output_mem_config), t_nan, input_grad.value(), output_mem_config, input_grad);
    where(
        queue_id,
        ttnn::logical_and(queue_id, ttnn::eqz(queue_id, input, output_mem_config), ttnn::eqz(queue_id, grad, output_mem_config), std::nullopt, output_mem_config),
        t_nan,
        input_grad.value(),
        output_mem_config,
        input_grad);

    result.emplace_back(input_grad);
    return result;
}

std::vector<std::optional<ttnn::Tensor>> ExecuteUnaryBackwardRsqrt::invoke(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    return ExecuteUnaryBackwardRsqrt::invoke(DefaultQueueId, grad, input, output_mem_config, input_grad);
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardNeg::invoke(uint8_t queue_id, const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt};
    input_grad = input_grad.value_or(ttnn::empty_like(input));
    result[0] = ttnn::neg(queue_id, grad, output_mem_config, input_grad);
    return result;
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardNeg::invoke(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    return ExecuteUnaryBackwardNeg::invoke(DefaultQueueId, grad, input, output_mem_config, input_grad);
}

std::vector<Tensor> _relu_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(ttnn::gtz(input, output_mem_config), grad, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// fill_bw:
// name: fill.Scalar(Tensor self, Scalar value) -> Tensor
// self: zeros_like(grad)
// result: at::fill(self_t, 0)
std::vector<std::optional<Tensor>> ExecuteUnaryBackwardFill::invoke(uint8_t queue_id, const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    std::vector<std::optional<Tensor>> result = {std::nullopt};
    result[0] = input_grad.has_value() ? ttnn::zeros_like(grad, std::nullopt, std::nullopt, std::nullopt, std::nullopt, input_grad) : ttnn::zeros_like(grad);
    return result;
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardFill::invoke(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    return ExecuteUnaryBackwardFill::invoke(DefaultQueueId, grad, input, output_mem_config, input_grad);
}

std::vector<Tensor> _hardsigmoid_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = ttnn::where(
        ttnn::logical_or(
            ttnn::le(input, -3, std::nullopt, output_mem_config),
            ttnn::ge(input, 3, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        ttnn::zeros_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config),
        ttnn::multiply(grad, 1.0 / 6),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// name: cos(Tensor self) -> Tensor
// self: grad * -self.sin()
std::vector<Tensor> _cos_bw(const Tensor& grad, const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result =
        ttnn::multiply(grad, (ttnn::neg(ttnn::sin(input_tensor, output_mem_config), output_mem_config)), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _acosh_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor in_rsqrt = ttnn::square(input, output_mem_config);
    in_rsqrt = ttnn::rsqrt(ttnn::subtract(in_rsqrt, 1.0, std::nullopt, output_mem_config), true, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, in_rsqrt, std::nullopt, output_mem_config);
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    Tensor cond_result = ttnn::logical_or(
        ttnn::lt(input, ttnn::full_like(input, -1.0f), std::nullopt, output_mem_config),
        ttnn::gt(input, ttnn::full_like(input, 1.0f), std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_a = ttnn::where(ttnn::eqz(cond_result, output_mem_config), t_nan, grad_a, output_mem_config);
    cond_result = ttnn::logical_or(
        ttnn::eq(input, ttnn::full_like(input, -1.0f), std::nullopt, output_mem_config),
        ttnn::eq(input, ttnn::full_like(input, 1.0f), std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_a = ttnn::where(
        ttnn::eq(cond_result, ttnn::full_like(input, 1.0f), std::nullopt, output_mem_config),
        t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// # - name: acos(Tensor self) -> Tensor
// #   self: grad * -((-self * self + 1).rsqrt())
std::vector<Tensor> _acos_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor neg_in = ttnn::neg(input, output_mem_config);
    Tensor in_rsqrt =
        ttnn::rsqrt(ttnn::add(ttnn::multiply(neg_in, input, std::nullopt, output_mem_config), 1.0f, std::nullopt, output_mem_config), true, output_mem_config);
    in_rsqrt = ttnn::neg(in_rsqrt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, in_rsqrt, std::nullopt, output_mem_config);
    Tensor neg_one = ttnn::full_like(input, -1.0f);
    Tensor pos_one = ttnn::full_like(input, 1.0f);
    Tensor t_inf = ttnn::multiply(ttnn::sign(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_or(
            ttnn::lt(input, neg_one, std::nullopt, output_mem_config),
            ttnn::gt(input, pos_one, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        std::nanf(" "),
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::eq(input, neg_one, std::nullopt, output_mem_config),
        t_inf,
        where(ttnn::eq(input, pos_one, std::nullopt, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> _atan_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::SQUARE},
    UnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam{UnaryOpType::RECIP}};
    Tensor grad_a = ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> _rad2deg_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float M_180_PI = 180 / M_PI;
    Tensor grad_result = ttnn::multiply(grad, M_180_PI, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _logit_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result =
        ttnn::multiply(grad,
            ttnn::reciprocal(ttnn::multiply(input, ttnn::rsub(input, 1.0f, output_mem_config), std::nullopt, output_mem_config)),
            std::nullopt,
            output_mem_config);
    Tensor status = ttnn::logical_and(
        ttnn::ge(input, 0.0f, std::nullopt, output_mem_config),
        ttnn::le(input, 1.0f, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = where(
        ttnn::eq(status, ttnn::ones_like(input), std::nullopt, output_mem_config), grad_result, std::nanf(""));
    grad_result = where(
        ttnn::logical_or(
            ttnn::eq(input, 0.0, std::nullopt, output_mem_config),
            ttnn::eq(input, 1.0, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config),
        grad_result,
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
// square
// result:  2 * input * grad_data
std::vector<Tensor> _square_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(ttnn::multiply(grad, 2.0f, std::nullopt, output_mem_config), input, std::nullopt, output_mem_config);
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
            ttnn::lt(input_tensor, ttnn::full_like(input_tensor, -lambd, input_tensor.get_dtype(), input_tensor.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config),
            ttnn::gt(input_tensor, ttnn::full_like(input_tensor, lambd, input_tensor.get_dtype(), input_tensor.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        grad,
        ttnn::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}


// Leaky_Relu
// result: torch.where(self > 0, grad_output, grad_output * negative_slope)
std::vector<Tensor> ExecuteUnaryBackwardLeakyRelu::invoke(
    const Tensor& grad, const Tensor& input, float negative_slope, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config), grad,  ttnn::multiply(grad, negative_slope, std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


// ELU
// result : grad * (torch.where(input >= 0, 1, alpha * torch.exp(input)))
std::vector<Tensor> ExecuteUnaryBackwardElu::invoke(
    const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::gez(input, output_mem_config),
        grad,
        ttnn::multiply(grad, ttnn::multiply(ttnn::exp(input, false, output_mem_config), alpha, std::nullopt, output_mem_config), std::nullopt, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


// Celu
// result: torch.where((input > 0), grad, grad * torch.exp(input / alpha))
std::vector<Tensor> ExecuteUnaryBackwardCelu::invoke(
    const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor div_result = ttnn::multiply(
        input, ttnn::reciprocal(ttnn::full_like(input, alpha, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    Tensor exp_result = ttnn::exp(div_result, false, output_mem_config);
    Tensor grad_result = where(
        ttnn::gt(input, ttnn::zeros_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config),
        grad,
        ttnn::multiply(grad, exp_result, std::nullopt, output_mem_config),
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


std::vector<Tensor> _rpow_bw(
    const Tensor& grad, const Tensor& input, float exponent, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    Tensor grad_result = ttnn::zeros_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    if (exponent != 0.0) {
        grad_result =
            ttnn::multiply(grad,
                ttnn::multiply(pow(input, exponent - 1, output_mem_config), exponent, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config);
        grad_result = ttnn::where(ltz(input, output_mem_config), t_nan, grad_result, output_mem_config);
    }
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


std::vector<Tensor> _floor_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = ttnn::zeros_like(grad);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}

std::vector<Tensor> _round_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = ttnn::zeros_like(grad);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}

std::vector<Tensor> _log_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor t_inf = ttnn::full_like(input, std::numeric_limits<float>::infinity());
    Tensor t_nan = ttnn::full_like(input, std::nanf(""));
    grad_tensor.emplace_back(where(
        ttnn::eqz(input, output_mem_config),
        where(
            ttnn::eqz(grad, output_mem_config),
            t_nan,
            ttnn::multiply(t_inf, ttnn::sign(grad, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        grad_a,
        output_mem_config));
    return grad_tensor;
}

std::vector<Tensor> _relu6_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_tensor = ttnn::zeros_like(input);
    Tensor one_tensor = ttnn::ones_like(input);
    Tensor six_tensor = ttnn::full_like(input, 6);
    Tensor grad_result =
        where(ttnn::le(input, zero_tensor, std::nullopt, output_mem_config), zero_tensor, six_tensor, output_mem_config);
    grad_result = where(
        ttnn::logical_and(
            ttnn::gtz(input, output_mem_config),
            ttnn::lt(input, six_tensor, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        grad,
        grad_result,
        output_mem_config);
    grad_result =
        where(ttnn::ge(input, six_tensor, std::nullopt, output_mem_config), zero_tensor, grad_result, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _abs_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(grad, ttnn::sign(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// Silu
// result:  grad * sigmoid_result * (1 + input * (1 - sigmoid_result))
std::vector<std::optional<Tensor>> ExecuteUnaryBackwardSilu::invoke(uint8_t queue_id, const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt};

    input_grad = input_grad.value_or(ttnn::empty_like(input));
    Tensor grad_sigmoid = ttnn::multiply(queue_id, grad, ttnn::sigmoid(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor add_sub = ttnn::add(queue_id,
        ttnn::multiply(queue_id, ttnn::subtract(queue_id, ttnn::full_like(input, 1.0f) , ttnn::sigmoid(input, output_mem_config), std::nullopt, output_mem_config),
            input,
            std::nullopt,
            output_mem_config),
        1.0f,
        std::nullopt,
        output_mem_config);
    ttnn::multiply(queue_id, grad_sigmoid, add_sub, std::nullopt, output_mem_config, input_grad);

    result[0] = input_grad;
    return result;
}

std::vector<std::optional<Tensor>> ExecuteUnaryBackwardSilu::invoke(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> input_grad) {
    return ExecuteUnaryBackwardSilu::invoke(DefaultQueueId, grad, input, output_mem_config, input_grad);
}

// Selu
// result:  torch.where(input > 0, grad * lambd, grad * lambd * alpha * torch.exp(input))
std::vector<Tensor> _selu_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_lambd = ttnn::multiply(grad, 1.0507f, std::nullopt, output_mem_config);
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config),
        grad_lambd,
        ttnn::multiply(ttnn::multiply(grad_lambd, 1.673260f, std::nullopt, output_mem_config),
            ttnn::exp(input, false, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// Hardswish
// result: torch.where(input < -3,0.0,torch.where(input <= 3, grad * ((input / 3) + 0.5), grad),)
std::vector<Tensor> _hardswish_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::lt(input, ttnn::full_like(input, -3.0f), std::nullopt, output_mem_config),
        0.0,
        where(
            ttnn::le(input, ttnn::full_like(input, 3.0f), std::nullopt, output_mem_config),
            ttnn::multiply(grad,
                ttnn::add(ttnn::multiply(input, 0.3333f, std::nullopt, output_mem_config), 0.5f, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            grad),
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// tanhshrink
// result:  torch.square(torch.tanh(input)) * grad_data
std::vector<Tensor> _tanhshrink_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tanh_res = ttnn::square(ttnn::tanh(input, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(ttnn::multiply(grad, tanh_res, std::nullopt, output_mem_config));
    return grad_tensor;
}

std::vector<Tensor> _atanh_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam {UnaryOpType::SQUARE},
    UnaryWithParam {UnaryOpType::SUB_UNARY_SFPU, 1.0f},
    UnaryWithParam {UnaryOpType::NEG},
    UnaryWithParam {UnaryOpType::RECIP}};

    Tensor grad_a =
        ttnn::multiply(grad, unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(ttnn::eqz(grad, output_mem_config), t_nan, grad_a, output_mem_config);
    grad_a = where(ttnn::logical_and(ttnn::eqz(grad, output_mem_config), ttnn::eqz(input, output_mem_config)), 0, grad_a, output_mem_config);
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
std::vector<Tensor> _asin_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam {UnaryOpType::SQUARE},
    UnaryWithParam {UnaryOpType::NEG},
    UnaryWithParam {UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam {UnaryOpType::RSQRT, true}};

    Tensor grad_result =
        ttnn::multiply(grad, unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    Tensor t_inf = ttnn::full_like(input, std::numeric_limits<float>::infinity());
    Tensor t_nan = ttnn::full_like(input, std::nanf(""));
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
std::vector<Tensor> _asinh_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam {UnaryOpType::SQUARE},
    UnaryWithParam {UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam {UnaryOpType::RSQRT, true}};
    Tensor grad_result =
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// name: sin(Tensor self) -> Tensor
// self: grad * self.cos()
std::vector<Tensor> _sin_bw(const Tensor& grad, const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_input = ttnn::multiply(grad, ttnn::cos(input_tensor, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_input);
    return grad_tensor;
}

// name: sinh(Tensor self) -> Tensor
// self: grad * self.cosh()
std::vector<Tensor> _sinh_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor grad_a = where(
        ttnn::gt(input, ttnn::full_like(input, 88.5f), std::nullopt, output_mem_config),
        t_inf,
        where(
            ttnn::lt(input, ttnn::full_like(input, -88.5f), std::nullopt, output_mem_config),
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
std::vector<Tensor> _log10_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor grad_a = ttnn::multiply(
        grad, ttnn::reciprocal(ttnn::multiply(input, M_LN10, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        std::nanf(" "),
        where(ttnn::eqz(input, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// bw(log1p(in)) = grad/(in + 1)
// for -1 = inf
std::vector<Tensor> _log1p_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor t_inp1 = ttnn::add(input, 1.0f, std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(t_inp1, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::eq(input, ttnn::full_like(input, -1.0f), std::nullopt, output_mem_config),
        t_inf,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(t_inp1, output_mem_config), eqz(grad, output_mem_config)),
        std::nanf(" "),
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> _erfc_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        ttnn::multiply(ttnn::exp(ttnn::neg(ttnn::square(input, output_mem_config), output_mem_config), false, output_mem_config),
            grad,
            std::nullopt,
            output_mem_config),
        -M_2_SQRTPI,
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _ceil_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = ttnn::zeros_like(grad);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}

// softsign
// result = grad_data / torch.square(1 + torch.abs(input))
std::vector<Tensor> _softsign_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam {UnaryOpType::ABS},
    UnaryWithParam {UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam {UnaryOpType::SQUARE},
    UnaryWithParam {UnaryOpType::RECIP}};
    grad_tensor.emplace_back(
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config));
    return grad_tensor;
}


// name: cosh(Tensor self) -> Tensor
// self: grad * self.sinh()
std::vector<Tensor> _cosh_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor t_neg_inf =
        ttnn::multiply(ttnn::sign(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor grad_a = where(
        ttnn::gt(input,
        ttnn::full_like(input, 88.50f, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config), t_inf,
        where(
            ttnn::lt(input,
            ttnn::full_like(input, -88.50f, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config),
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
    Tensor grad_result =
        ttnn::multiply(grad,
            ttnn::reciprocal(ttnn::multiply(input, ttnn::rsub(input, 1.0f, output_mem_config), std::nullopt, output_mem_config)),
            std::nullopt,
            output_mem_config);
    Tensor t_eps = ttnn::full_like(input, eps, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    Tensor t_low = ttnn::full_like(input, low, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    Tensor t_high = ttnn::full_like(input, high, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    Tensor ltl_gth = ttnn::logical_or(
        ttnn::lt(input, t_low, std::nullopt, output_mem_config),
        ttnn::gt(input, t_high, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = where(
        ttnn::eq(ltl_gth, ttnn::ones_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config),
        where(ttnn::ltz(t_eps, output_mem_config), std::nanf(" "), 0.0, output_mem_config),
        where(
            ttnn::logical_or(
                ttnn::eq(input, 0.0, std::nullopt, output_mem_config),
                ttnn::eq(input, 1.0, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config),
            grad_result,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


// bw(log2(in)) = grad/(in * 0.69314718055994530942)
std::vector<Tensor> _log2_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor grad_a = ttnn::multiply(
        grad, ttnn::reciprocal(ttnn::multiply(input, M_LN2, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        std::nanf(" "),
        where(ttnn::eqz(input, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}


std::vector<Tensor> _sign_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = ttnn::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}


std::vector<Tensor> _div_no_nan_bw(
    const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zeros = ttnn::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
    Tensor val = ttnn::full_like(input, scalar, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    Tensor result = where(
        ttnn::eq(val, 0, std::nullopt, output_mem_config), zeros, ttnn::multiply(grad, 1 / scalar, std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// #  bw (exp2) = grad * exp2(input) * M_LN2
// # M_LN2 = 0.693147180559945309417
std::vector<Tensor> _exp2_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor exp_result = ttnn::exp2(input, output_mem_config);
    exp_result = ttnn::multiply(exp_result, M_LN2, std::nullopt, output_mem_config);
    Tensor result = ttnn::multiply(grad, exp_result, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// bw(expm1) = grad * expm1(input) + 1
std::vector<Tensor> _expm1_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
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
    Tensor t_inf = ttnn::full_like(input, std::numeric_limits<float>::infinity());
    Tensor t_nan = ttnn::full_like(input, std::nanf(""));
    grad_tensor.emplace_back(where(
        ttnn::eqz(input, output_mem_config),
        where(
            ttnn::eqz(grad, output_mem_config),
            t_nan,
            ttnn::multiply(t_inf, ttnn::neg(ttnn::sign(grad, output_mem_config), output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        ttnn::multiply(ttnn::neg(grad, output_mem_config),
            ttnn::reciprocal(ttnn::square(input, output_mem_config), output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config));
    return grad_tensor;
}

std::vector<ComplexTensor> ExecuteUnaryBackwardRecip::invoke(
    const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor condition_nan = ttnn::logical_and(ttnn::eqz(input.real(),output_mem_config), ttnn::eqz(input.imag(),output_mem_config), std::nullopt, output_mem_config);
    ComplexTensor neg_grad = ComplexTensor({ttnn::neg(grad.real(),output_mem_config), ttnn::neg(grad.imag(),output_mem_config)});
    ComplexTensor inp_recip = ttnn::reciprocal(input, output_mem_config);
    ComplexTensor grad_inp = ttnn::operations::complex_binary::_mul(neg_grad, ttnn::conj(ttnn::operations::complex_binary::_mul(inp_recip, inp_recip, output_mem_config), output_mem_config), output_mem_config) ;
    neg_grad.deallocate();
    inp_recip.deallocate();
    Tensor grad_inp_r = where(condition_nan, ttnn::operations::creation::full_like(input.real(), std::nanf(""), std::nullopt, std::nullopt, std::nullopt, output_mem_config), grad_inp.real(), output_mem_config);
    Tensor grad_inp_i = where(condition_nan, ttnn::operations::creation::full_like(input.imag(), std::nanf(""), std::nullopt, std::nullopt, std::nullopt, output_mem_config), grad_inp.imag(), output_mem_config);
    condition_nan.deallocate();
    grad_inp = ComplexTensor({ grad_inp_r, grad_inp_i});
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
    Tensor grad_inp_r = where(ttnn::eqz(result, output_mem_config), ttnn::zeros_like(result, result.get_dtype(), result.get_layout(), std::nullopt, output_mem_config), ttnn::multiply(grad, ttnn::multiply(input.real(), ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),std::nullopt, output_mem_config), output_mem_config );
    Tensor grad_inp_i = where(ttnn::eqz(result, output_mem_config), ttnn::zeros_like(result, result.get_dtype(), result.get_layout(), std::nullopt, output_mem_config), ttnn::multiply(grad, ttnn::multiply(input.imag(), ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),std::nullopt, output_mem_config), output_mem_config );
    ComplexTensor grad_inp = ComplexTensor({ grad_inp_r, grad_inp_i});
    result.deallocate();
    grad_inp_r.deallocate();
    grad_inp_i.deallocate();
    grad_tensor.emplace_back(grad_inp);
    return grad_tensor;
}


std::vector<Tensor> _digamma_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");
    Tensor grad_a = ttnn::multiply(grad, ttnn::polygamma(input, 1, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_nan,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
        -t_inf,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> _polygamma_bw(
    const Tensor& grad, const Tensor& input, int n, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    float t_nan = std::nanf("");
    float pos_neg = 1.0f;
    if (n == 2 || n == 4 || n == 6 || n == 8 || n == 10) {
        pos_neg = -1.0f;
    }
    Tensor grad_a = ttnn::multiply(grad, ttnn::polygamma(input, (n + 1), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::le(input, 0.0, std::nullopt, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_nan,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
        ttnn::multiply(
            ttnn::full_like(input, -std::numeric_limits<float>::infinity(), std::nullopt, std::nullopt, std::nullopt, output_mem_config), pos_neg, std::nullopt, output_mem_config),
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
        ttnn::multiply(
            ttnn::full_like(input, std::numeric_limits<float>::infinity(), std::nullopt, std::nullopt, std::nullopt, output_mem_config), pos_neg, std::nullopt, output_mem_config),
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// erfinv
// self: 0.5 * sqrt(M_PI) * exp(self.erfinv().pow(2)) * grad
// for input -1 and 1: grad.sign() * inf, for input > 1 or < -1 : nan
std::vector<Tensor> _erfinv_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        ttnn::multiply(ttnn::sqrt(ttnn::operations::creation::full_like(input, M_PI), output_mem_config),
            ttnn::multiply(ttnn::exp(ttnn::square(ttnn::erfinv(input, output_mem_config), output_mem_config), false, output_mem_config),
                grad,
                std::nullopt,
                output_mem_config),
            std::nullopt,
            output_mem_config),
        0.5,
        std::nullopt,
        output_mem_config);
    Tensor neg_one = ttnn::full_like(input, -1.0f);
    Tensor pos_one = ttnn::full_like(input, 1.0f);
    Tensor t_inf = ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    result = ttnn::where(
        ttnn::logical_or(
            ttnn::lt(input, neg_one, std::nullopt, output_mem_config),
            ttnn::gt(input, pos_one, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        std::nanf(" "),
        result,
        output_mem_config);
    result = ttnn::where(
        ttnn::eq(input, neg_one, std::nullopt, output_mem_config),
        t_inf,
        ttnn::where(ttnn::eq(input, pos_one, std::nullopt, output_mem_config), t_inf, result, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _erf_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        ttnn::multiply(ttnn::exp(ttnn::neg(ttnn::square(input, output_mem_config), output_mem_config), false, output_mem_config),
            grad,
            std::nullopt,
            output_mem_config),
        M_2_SQRTPI,
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _deg2rad_bw(const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float M_PI_180 = M_PI / 180;
    Tensor grad_result = ttnn::multiply(grad, M_PI_180, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


std::vector<std::optional<ttnn::Tensor>> ExecuteUnaryBackwardGelu::invoke(
    uint8_t queue_id,
    const Tensor& grad,
    const Tensor& input,
    string approximate,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result;
    if(!input_grad.has_value()){
        input_grad = ttnn::empty_like(grad);
    }

    auto output_memory_config = output_mem_config.value_or(input.memory_config()); //TODO: Remove after ternary forward ops migration is completed
    TT_FATAL((approximate == "none" || approximate == "tanh"), "Incorrect approximate mode (expected 'None', 'tanh')");

    if (approximate == "tanh") {
        float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
        float kKappa = 0.044715;
        Tensor x_sq = ttnn::multiply(input, input, std::nullopt, output_memory_config);
        Tensor x_cube = ttnn::multiply(x_sq, input, std::nullopt, output_memory_config);
        Tensor inner = ttnn::multiply(ttnn::add(input, ttnn::multiply(x_cube, kKappa, std::nullopt, output_memory_config)), kBeta, std::nullopt, output_mem_config);
        Tensor tanh_inner = ttnn::tanh(inner, output_memory_config);

        Tensor left = ttnn::multiply(input, 0.5, std::nullopt, output_memory_config);
        Tensor right = ttnn::add(tanh_inner, 1, std::nullopt, output_memory_config);

        Tensor left_derivative = ttnn::multiply(right, 0.5, std::nullopt, output_memory_config);

        Tensor tanh_derivative =
            ttnn::neg(ttnn::subtract(ttnn::multiply(tanh_inner, tanh_inner, std::nullopt, output_memory_config), 1, std::nullopt, output_mem_config),
                output_memory_config);
        Tensor inner_derivative = ttnn::multiply(
            (ttnn::add(
                ttnn::multiply(ttnn::multiply(x_sq, kKappa, std::nullopt, output_memory_config), 3, std::nullopt, output_memory_config), 1, std::nullopt, output_mem_config)), kBeta);
        Tensor right_derivative =
            ttnn::multiply(ttnn::multiply(tanh_derivative, left, std::nullopt, output_memory_config),
                inner_derivative,
                std::nullopt,
                output_memory_config);

        ttnn::multiply(queue_id, grad, (ttnn::add(left_derivative, right_derivative)), std::nullopt, output_memory_config, input_grad);
        result.push_back(input_grad);
    } else {
        float kAlpha = M_SQRT1_2;
        float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
        Tensor cdf =
            ttnn::multiply((ttnn::add(ttnn::erf(ttnn::multiply(input, kAlpha, std::nullopt, output_memory_config)), 1, std::nullopt, output_memory_config)), 0.5);
        Tensor pdf = ttnn::multiply(ttnn::exp(ttnn::multiply(ttnn::multiply(input, input), -0.5), false, output_memory_config), kBeta, std::nullopt, output_memory_config);
        ttnn::multiply(queue_id, grad, ttnn::add(cdf, ttnn::multiply(input, pdf)), std::nullopt, output_memory_config, input_grad);
        result.push_back(input_grad);
    }

    return result;
}

std::vector<std::optional<ttnn::Tensor>> ExecuteUnaryBackwardGelu::invoke(
    const Tensor& grad,
    const Tensor& input,
    string approximate,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
        return ExecuteUnaryBackwardGelu::invoke(DefaultQueueId, grad, input, approximate, output_mem_config, input_grad);
}

std::vector<Tensor> ExecuteUnaryBackwardRepeat::invoke(
    const Tensor& grad, const Tensor& input, const tt::tt_metal::LegacyShape& shape, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config()); //TODO: Remove after ternary forward ops migration is completed

    auto shape_wh = input.get_legacy_shape();
    TT_FATAL(shape_wh[0] == 1 && "input shape[0] should be 1", "Error");
    auto ttnn_device = input.device();
    // input.get_legacy_shape()[0]
    // If repeat shape has 0's, it returns zeros of given input
    if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0 || shape[3] == 0) {
        Tensor zero_tensor = ttnn::zeros_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_memory_config);
        grad_tensor.emplace_back(zero_tensor);
        return grad_tensor;
    } else if (shape[0] > 1) {
        std::vector<int64_t> dim = {0};
        TT_FATAL(shape[1] == 1 && shape[2] == 1 && shape[3] == 1, "repeat[1], [2], [3] should be 1");
        std::array<std::uint32_t, 4> intended_shape_array = {1, shape_wh[1], shape_wh[2], shape_wh[3]};
        const ttnn::Shape required = ttnn::Shape(intended_shape_array);
        Tensor result = tt::operations::primary::moreh_sum(
            grad,
            dim,
            true,
            ttnn::zeros(required, input.get_dtype(), input.get_layout(), std::optional<std::reference_wrapper<tt::tt_metal::Device>>(*ttnn_device), output_memory_config),
            output_memory_config);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    } else if (shape[1] > 1) {
        std::vector<int64_t> dim = {1};
        TT_FATAL(shape[0] == 1 && shape[2] == 1 && shape[3] == 1, "repeat[0], [2], [3] should be 1");
        std::array<std::uint32_t, 4> intended_shape_array = {shape_wh[0], 1, shape_wh[2], shape_wh[3]};
        const ttnn::Shape required = ttnn::Shape(intended_shape_array);
        Tensor result = tt::operations::primary::moreh_sum(
            grad,
            dim,
            true,
            ttnn::zeros(required, input.get_dtype(), input.get_layout(), std::optional<std::reference_wrapper<tt::tt_metal::Device>>(*ttnn_device), output_memory_config),
            output_memory_config);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    }
    return grad_tensor;
}

// Autoformat support
Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& output_mem_config) {
    auto formatted_input_tensor = temp;
    if(formatted_input_tensor.get_layout()==Layout::ROW_MAJOR){
        auto a_pad_shape = ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(temp.get_legacy_shape(), false, false, true, true);
        if (!ttnn::operations::experimental::auto_format::AutoFormat::check_input_tensor_format(temp, a_pad_shape)) {
            formatted_input_tensor = ttnn::operations::experimental::auto_format::AutoFormat::format_input_tensor(temp, temp.device(), a_pad_shape, 1.0, Layout::TILE);
        }
    }
    return formatted_input_tensor;
}

// Prod
// along a single dimension --> result: grad_data * (y / input )
std::vector<Tensor> ExecuteUnaryBackwardProd::invoke(
    const Tensor& grad, const Tensor& input, bool all_dimensions, int64_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config()); //TODO: Remove after ternary forward ops migration is completed
    Tensor prod_result = ttnn::prod(input, all_dimensions, dim, output_memory_config);
    if(prod_result.get_layout()==Layout::ROW_MAJOR && prod_result.storage_type() == StorageType::DEVICE){
        prod_result = ttnn::operations::unary_backward::change_layout_to_tile(prod_result, output_memory_config);
        }
    if (all_dimensions == true) {
        Tensor temp =
            ttnn::multiply(prod_result, grad, std::nullopt, output_memory_config);  // result is stored in the first position
        Tensor fill_tensor = tt::numpy::fill_first_val_into_tensor<::bfloat16>(
            temp, temp.get_dtype(), temp.get_layout(), temp.device(), output_memory_config);
        Tensor all_dimension_result =
            ttnn::multiply(ttnn::reciprocal(input, output_memory_config), fill_tensor, std::nullopt, output_memory_config);
        grad_tensor.emplace_back(all_dimension_result);
        return grad_tensor;
    }
    // all_dimensions = False
    Tensor updated_grad = prod_result;
    auto step = std::vector<uint32_t>({1, 1, 1, 1});
    if (prod_result.get_legacy_shape().without_padding() != grad.get_legacy_shape()) {
        if (dim == 3 || dim == -1) {
            std::vector<int64_t> after_permute_dims = {0, 3, 1, 2};
            Tensor required = ttnn::permute(grad, after_permute_dims, output_memory_config);
            std::vector<uint32_t> start_index = {0, 0, 0, 0};
            std::vector<uint32_t> end_index = {
            grad.get_legacy_shape()[0], 1, grad.get_legacy_shape()[1], grad.get_legacy_shape()[2]};
            Tensor new_slice_tensor = ttnn::slice(DefaultQueueId, required, start_index, end_index, step, std::nullopt);
            after_permute_dims = {0, 2, 3, 1};
            updated_grad = ttnn::permute(new_slice_tensor, after_permute_dims, output_memory_config);
            if(updated_grad.storage_type() != StorageType::DEVICE && updated_grad.storage_type() != StorageType::MULTI_DEVICE) {
                Tensor pad_updated_grad = updated_grad.pad_to_tile(1.0f);
                pad_updated_grad = pad_updated_grad.to(Layout::TILE);
                updated_grad = pad_updated_grad.to(input.device());
            }
        } else if (dim == 2 || dim == -2) {
            std::vector<int64_t> after_permute_dims = {0, 2, 1, 3};
            Tensor required = ttnn::permute(grad, after_permute_dims, output_memory_config);
            std::vector<uint32_t> start_index = {0, 0, 0, 0};
            std::vector<uint32_t> end_index = {
            grad.get_legacy_shape()[0], 1, grad.get_legacy_shape()[1], grad.get_legacy_shape()[3]};
            Tensor new_slice_tensor = ttnn::slice(DefaultQueueId, required, start_index, end_index, step, std::nullopt);
            updated_grad = ttnn::permute(new_slice_tensor, after_permute_dims, output_memory_config);
            if(updated_grad.get_layout()==Layout::ROW_MAJOR){
                updated_grad = ttnn::operations::unary_backward::change_layout_to_tile(updated_grad, output_memory_config);
            }
        }
    }
    Tensor reciprocal_input = ttnn::reciprocal(input, output_memory_config);
    Tensor temp = ttnn::multiply(prod_result, (dim == 1 || dim == 0 || dim == -4 || dim == -3) ? grad : updated_grad, std::nullopt, output_memory_config);
    if(temp.get_layout()==Layout::ROW_MAJOR){
        temp = ttnn::operations::unary_backward::change_layout_to_tile(temp, output_memory_config);
    }
    if (dim == 3 || dim == -1) {
        Tensor grad_result = ttnn::bcast(0, reciprocal_input, temp, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::W, output_memory_config);
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    } else if (dim == 2 || dim == -2) {
        Tensor grad_result = ttnn::bcast(0, reciprocal_input, temp, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::H, output_memory_config);
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    } else if (dim == 1 || dim == -3) {
        Tensor tensor_1_temp = reciprocal_input;
        if (reciprocal_input.get_legacy_shape()[1] % 32 != 0) {
            std::vector<std::pair<uint32_t, uint32_t>> padding = {{0, 0},
                          {0, 32 - (reciprocal_input.get_legacy_shape()[1] % 32)},
                          {0, 0},
                          {0, 0}};
            tensor_1_temp = ttnn::pad(0, reciprocal_input, padding, 0, true, std::nullopt);
        }
        std::vector<int64_t> after_permute_dims = {0, 2, 3, 1};
        Tensor tensor_1 = ttnn::permute(tensor_1_temp, after_permute_dims, output_memory_config);
        Tensor tensor_2 = ttnn::permute(temp, after_permute_dims, output_memory_config);

        // put the tensor back on device because permute throws it off device
        // See: Remove auto format within permute_op.cpp #9404
        tensor_2 = ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_device_and_pad(tensor_2, tensor_1.device(),tensor_1.get_layout(), tensor_1.memory_config());

        after_permute_dims = {0, 3, 1, 2};
        Tensor result = permute(
            ttnn::bcast(0, tensor_1, tensor_2, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::W, output_memory_config),
            after_permute_dims,
            output_memory_config);
        Tensor grad_result = result;
        if (reciprocal_input.get_legacy_shape()[1] % 32 != 0) {
            std::vector<uint32_t> start_index = {0, 0, 0, 0};
            std::vector<uint32_t> end_index = {
                input.get_legacy_shape()[0],
                input.get_legacy_shape()[1],
                input.get_legacy_shape()[2],
                input.get_legacy_shape()[3]};
            auto step = std::vector<uint32_t>({1,1,1,1});
            grad_result = ttnn::slice(DefaultQueueId, result, start_index, end_index, step, std::nullopt);
        }
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    }
    // dim 0
    Tensor tensor_1_temp = reciprocal_input;
    if (reciprocal_input.get_legacy_shape()[0] % 32 != 0) {
        std::vector<std::pair<uint32_t, uint32_t>> padding = {{0, (32 - (reciprocal_input.get_legacy_shape()[0] % 32))},
                      {0, 0},
                      {0, 0},
                      {0, 0}};
        tensor_1_temp = ttnn::pad(0, reciprocal_input, padding, 0, false, std::nullopt);
    }
    std::vector<int64_t> after_permute_dims = {3, 1, 2, 0};
    Tensor tensor_1 = ttnn::permute(tensor_1_temp, after_permute_dims, output_memory_config);
    Tensor tensor_2 = ttnn::permute(temp, after_permute_dims, output_memory_config);

    // put the tensor back on device because permute throws it off device
    // See: Remove auto format within permute_op.cpp #9404
    tensor_2 = ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_device_and_pad(tensor_2, tensor_1.device(),tensor_1.get_layout(), tensor_1.memory_config());

    Tensor result = ttnn::permute(
        ttnn::bcast(0, tensor_1, tensor_2, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::W, output_memory_config),
        after_permute_dims,
        output_memory_config);
    Tensor grad_result = result;
    if (reciprocal_input.get_legacy_shape()[0] % 32 != 0) {
        std::vector<uint32_t> start_index = {0, 0, 0, 0};
        std::vector<uint32_t> end_index = {
            input.get_legacy_shape()[0],
            input.get_legacy_shape()[1],
            input.get_legacy_shape()[2],
            input.get_legacy_shape()[3]};
        grad_result = ttnn::slice(DefaultQueueId, result, start_index, end_index, step, std::nullopt);
    }
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

}  // namespace ttnn::operations::unary
