// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/decorators.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"

#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary_backward/device/binary_backward_op.hpp"
#include "ttnn/operations/eltwise/unary_backward/device/unary_backward_op.hpp"

#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_composite_op.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/operations/creation.hpp"

#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::binary_backward {

std::vector<ttnn::Tensor> _atan2_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam {UnaryOpType::SQUARE},
    UnaryWithParam {UnaryOpType::RECIP}};
    Tensor recip_mul =
        ttnn::multiply(grad, ttnn::unary_chain(ttnn::hypot(input, other, output_mem_config), ops_chain, output_mem_config), std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(other, recip_mul, std::nullopt, output_mem_config);
    Tensor cond = ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::eqz(other, output_mem_config));
    grad_a = ttnn::where(cond, t_nan, grad_a, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = ttnn::multiply(ttnn::neg(input), recip_mul, std::nullopt, output_mem_config);
    grad_b = ttnn::where(cond, t_nan, grad_b, output_mem_config);
    recip_mul.deallocate();
    cond.deallocate();
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<std::optional<ttnn::Tensor>> _addalpha_bw(
    uint8_t queue_id,
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result;

    if (are_required_outputs.at(0)) {
        if(input_grad.has_value()){
            ttnn::assign(queue_id, grad, input_grad.value());
        } else {
            input_grad = grad;
        }
        result.emplace_back(input_grad);
    } else {
        result.emplace_back(std::nullopt);
    }
    if (are_required_outputs.at(1)) {
        if(other_grad.has_value()){
            ttnn::multiply(queue_id, grad, ttnn::operations::creation::full_like(grad, alpha, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config), std::nullopt, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, other_grad);
        } else {
            other_grad = ttnn::multiply(queue_id, grad, alpha, std::nullopt, output_mem_config);
        }
        result.emplace_back(other_grad);
    } else {
        result.emplace_back(std::nullopt);
    }

    return std::move(result);

}

std::vector<ttnn::Tensor> _subalpha_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_b = ttnn::multiply(ttnn::neg(grad, output_mem_config), alpha, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<ttnn::Tensor> _sub_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    return _subalpha_bw(grad, input, other, 1.0, output_mem_config);
}

std::vector<std::optional<Tensor>> _add_bw(
    uint8_t queue_id,
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    return _addalpha_bw(queue_id, grad, input, other, 1.0f, output_mem_config, are_required_outputs, input_grad, other_grad);
}

std::vector<Tensor> ExecuteBackwardAdd::invoke(
    const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> ExecuteBackwardAdd::invoke(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_b = ttnn::multiply(grad, 1.0f, std::nullopt,output_memory_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<ComplexTensor> ExecuteBackwardAdd::invoke(
    const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a = grad;
    grad_tensor.emplace_back(grad_a);
    const Tensor& grad_r = grad.real();
    const Tensor& grad_i = grad.imag();
    ComplexTensor grad_b = ComplexTensor({ttnn::multiply(grad_r, alpha, std::nullopt, output_mem_config), ttnn::multiply(grad_i, alpha, std::nullopt, output_mem_config)});
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<Tensor> ExecuteBackwardSub::invoke(
    const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> ExecuteBackwardSub::invoke(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_b = ttnn::multiply(ttnn::neg(grad, output_mem_config), 1.0f, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<ComplexTensor> ExecuteBackwardSub::invoke(
    const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a = grad;
    grad_tensor.emplace_back(grad);
    const Tensor& grad_r = grad.real();
    const Tensor& grad_i = grad.imag();
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::NEG},
    UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, alpha} };
    ComplexTensor grad_b = ComplexTensor({ttnn::unary_chain( grad_r, ops_chain, output_mem_config), ttnn::unary_chain( grad_i, ops_chain, output_mem_config)});
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<ttnn::Tensor> _xlogy_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad1_result = ttnn::log(other, output_mem_config);
    Tensor zero_tensor = ttnn::zeros_like(other, other.get_dtype(), other.get_layout(), std::nullopt, output_mem_config);
    grad1_result = ttnn::where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config),
            ttnn::le(other, zero_tensor, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        zero_tensor,
        ttnn::where(ttnn::ltz(other, output_mem_config), std::nanf(" "), grad1_result, output_mem_config),
        output_mem_config);
    grad1_result =
        ttnn::where(ttnn::eq(input, std::nanf(" "), std::nullopt, output_mem_config), std::nanf(" "), grad1_result, output_mem_config);
    grad1_result = ttnn::multiply(grad, grad1_result, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad1_result);
    Tensor div_result = ttnn::multiply(input, ttnn::reciprocal(other, output_mem_config), std::nullopt, output_mem_config);
    Tensor grad2_result = ttnn::multiply(grad, div_result, std::nullopt, output_mem_config);
    grad2_result = where(
        ttnn::eqz(other, output_mem_config),
        ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config),
        grad2_result,
        output_mem_config);
    grad2_result =
        ttnn::where(ttnn::eq(other, std::nanf(" "), std::nullopt, output_mem_config), std::nanf(" "), grad2_result, output_mem_config);
    grad_tensor.emplace_back(grad2_result);
    return grad_tensor;
}


std::vector<ttnn::Tensor> _hypot_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    Tensor result_recip = ttnn::reciprocal(ttnn::hypot(input, other, output_memory_config), output_memory_config);
    Tensor grad_a =
        ttnn::multiply(grad, ttnn::multiply(input, result_recip, std::nullopt, output_memory_config), std::nullopt, output_memory_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b =
        ttnn::multiply(grad, ttnn::multiply(other, result_recip, std::nullopt, output_memory_config), std::nullopt, output_memory_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}


// torch reference
// - name: ldexp(Tensor self, Tensor other) -> Tensor
//   self: grad * 2^other
//   other: grad * self * ln(2) * (2^other)
// # M_LN2 = ln(2)= 0.693147180559945309417
std::vector<ttnn::Tensor> _ldexp_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    Tensor tpow_o = ttnn::multiply(grad, ttnn::rpow(other, 2.0, output_memory_config), std::nullopt, output_memory_config);
    grad_tensor.emplace_back(tpow_o);
    Tensor result = ttnn::multiply(input, ttnn::multiply(tpow_o, M_LN2, std::nullopt, output_memory_config), std::nullopt, output_memory_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}


/*
Torch Reference:
name: logaddexp(Tensor self, Tensor other) -> Tensor
self: grad / (1 + exp(other - self)).conj()
other: grad / (1 + exp(self - other)).conj()
*/
std::vector<ttnn::Tensor> _logaddexp_bw(
    const Tensor& grad, const Tensor& input_a, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor opexp =
        ttnn::add(ttnn::exp(ttnn::subtract(other, input_a, std::nullopt, output_mem_config), false, output_mem_config), 1, std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(opexp, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    opexp = ttnn::add(ttnn::exp(ttnn::subtract(input_a, other, std::nullopt, output_mem_config), false, output_mem_config), 1, std::nullopt, output_mem_config);
    Tensor grad_b = ttnn::multiply(grad, ttnn::reciprocal(opexp, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

/*
Torch reference
name: logaddexp2(Tensor self, Tensor other) -> Tensor
self: grad / (1 + pow(2, other - self))
other: grad / (1 + pow(2, self - other))
*/

std::vector<ttnn::Tensor> _logaddexp2_bw(
    const Tensor& grad, const Tensor& input_a, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input_a.memory_config());
    Tensor oppow =
        ttnn::add(ttnn::rpow(ttnn::subtract(other, input_a, std::nullopt, output_memory_config), 2, output_memory_config), 1, std::nullopt, output_memory_config);
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(oppow, output_memory_config), std::nullopt, output_memory_config);
    grad_tensor.emplace_back(grad_a);
    oppow = ttnn::add(ttnn::rpow(ttnn::subtract(input_a, other, std::nullopt, output_memory_config), 2, output_memory_config), 1, std::nullopt, output_memory_config);
    Tensor grad_b = ttnn::multiply(grad, ttnn::reciprocal(oppow, output_memory_config), std::nullopt, output_memory_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}


std::vector<ttnn::Tensor> _squared_difference_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor difference = ttnn::subtract(input, other);
    Tensor grad_a = ttnn::multiply(ttnn::multiply(grad, difference, std::nullopt, output_mem_config), 2, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = ttnn::multiply(grad_a, -1, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> _eq_bw(
    uint8_t cq_id,
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result;

    if (are_required_outputs.at(0)) {
        input_grad = ttnn::full_like(input, 0.0f);
        result.emplace_back(input_grad);
    } else {
        result.emplace_back(std::nullopt);
    }
    if (are_required_outputs.at(1)) {
        other_grad = ttnn::full_like(grad, 0.0f);
        result.emplace_back(other_grad);
    } else {
        result.emplace_back(std::nullopt);
    }
    return result;
}

std::vector<ttnn::Tensor> _eq_bw_inter(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    auto result = _eq_bw(0, grad, input, other, output_mem_config, {true, true}, std::nullopt, std::nullopt);
    std::vector<ttnn::Tensor> output_tensors;
    output_tensors.reserve(result.size());

    for (const auto& opt_tensor : result) {
        if (opt_tensor) {
            output_tensors.emplace_back(*opt_tensor);
        } else {
            output_tensors.emplace_back();
        }
    }
    return output_tensors;
}

std::vector<Tensor> _assign_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _concat_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, int dim, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    std::vector<uint32_t> start_index = {0, 0, 0, 0};
    std::vector<uint32_t> end_index = {
        input.get_legacy_shape()[0] - 1,
        input.get_legacy_shape()[1] - 1,
        input.get_legacy_shape()[2] - 1,
        input.get_legacy_shape()[3] - 1};

    Tensor grad_a = ttnn::slice(0, grad, start_index, end_index, std::nullopt);
    grad_tensor.emplace_back(grad_a);

    std::vector<uint32_t> start_index_2 = {0, 0, 0, 0};
    if (dim == 0) {
        start_index_2 = {input.get_legacy_shape()[0], 0, 0, 0};
    } else if (dim == 1) {
        start_index_2 = {0, input.get_legacy_shape()[1], 0, 0};
    } else if (dim == 2) {
        start_index_2 = {
            0, 0, input.get_legacy_shape()[2], 0};
    } else if (dim == 3) {
        start_index_2 = {0, 0, 0, input.get_legacy_shape()[3]};
    }
    std::vector<uint32_t> end_index_2 = {
        grad.get_legacy_shape()[0] - 1,
        grad.get_legacy_shape()[1] - 1,
        grad.get_legacy_shape()[2] - 1,
        grad.get_legacy_shape()[3] - 1};
    Tensor grad_b = ttnn::slice(0, grad, start_index_2, end_index_2, std::nullopt);
    grad_tensor.emplace_back(grad_b);

    return grad_tensor;
}

std::vector<Tensor> _binary_comp_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = ttnn::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    Tensor zero_input = ttnn::zeros_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(zero_input);
    return grad_tensor;
}

std::vector<Tensor> _rsub_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor = _subalpha_bw(grad, input, other, 1.0f, output_mem_config.value_or(input.memory_config()));
    std::swap(grad_tensor[0], grad_tensor[1]);
    return grad_tensor;
}

std::vector<Tensor> ExecuteBackwardBiasGelu::invoke(
    const Tensor& grad, const Tensor& input_a, const Tensor& input_b, string approximate, const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL((approximate == "none" || approximate == "tanh") && "Incorrect approximation type (expected 'none', 'tanh')");
    std::vector<Tensor> grad_tensor;
    Tensor input = ttnn::add(input_a, input_b);
    grad_tensor = ttnn::gelu_bw(grad, input, approximate = approximate, output_mem_config);
    grad_tensor.emplace_back(grad_tensor[0]);
    return grad_tensor;
}

std::vector<Tensor> ExecuteBackwardBiasGelu::invoke(
    const Tensor& grad, const Tensor& input_tensor, float bias, string approximate, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    TT_FATAL((approximate == "none" || approximate == "tanh") && "Incorrect rounding mode (expected 'none' or 'tanh')");
    Tensor input = ttnn::add(input_tensor, bias);
    grad_tensor = ttnn::gelu_bw(grad, input, approximate = approximate);
    return grad_tensor;
}


std::vector<Tensor> _gt_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    return _binary_comp_bw(grad, input, other, output_mem_config);
}

std::vector<Tensor> _ge_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    return _binary_comp_bw(grad, input, other, output_mem_config);
}
std::vector<Tensor> _lt_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    return _binary_comp_bw(grad, input, other, output_mem_config);
}

// template parameter min_or_max = TRUE for MAX, FALSE for MIN
template <bool min_or_max>
std::vector<Tensor> _min_or_max_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor zeros_t = ttnn::zeros_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    std::vector<Tensor> grad_tensor;
    Tensor t_scale_grad = ttnn::multiply(grad, 0.5, std::nullopt, output_mem_config);
    Tensor t_sub = ttnn::subtract(other, input, std::nullopt, output_mem_config);
    Tensor t_sub_gtz = ttnn::gtz(t_sub, output_mem_config);
    Tensor t_sub_eqz = ttnn::eqz(t_sub, output_mem_config);
    Tensor t_sub_ltz = ttnn::ltz(t_sub, output_mem_config);
    Tensor grad_other =
        ttnn::add(ttnn::multiply(t_sub_ltz, grad, std::nullopt, output_mem_config),
            ttnn::multiply(t_sub_eqz, t_scale_grad, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config);
    Tensor grad_input =
        ttnn::add(ttnn::multiply(t_sub_gtz, grad, std::nullopt, output_mem_config),
            ttnn::multiply(t_sub_eqz, t_scale_grad, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config);

    if (min_or_max) {
        // MAX
        grad_tensor.emplace_back(grad_other);
        grad_tensor.emplace_back(grad_input);
    } else {
        // MIN
        grad_tensor.emplace_back(grad_input);
        grad_tensor.emplace_back(grad_other);
    }
    return grad_tensor;
}

template std::vector<Tensor> _min_or_max_bw<true>(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);

template std::vector<Tensor> _min_or_max_bw<false>(
    const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);


std::vector<Tensor> ExecuteBackwardDiv::invoke(
    const Tensor& grad, const Tensor& input, float scalar, std::string round_mode, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    TT_FATAL((round_mode == "None" || round_mode == "trunc" || round_mode == "floor") && "Incorrect rounding mode (expected 'None', 'trunc', or 'floor')");
    float inv_scalar = 1.0f / scalar;
    if (round_mode == "None") {
        Tensor t_inf = ttnn::operations::creation::full_like(input, std::numeric_limits<float>::infinity());
        if (scalar == 0.0) {
            float t_nan = std::nanf("");
            grad_tensor.emplace_back(where(
                ttnn::eqz(grad, output_mem_config),
                t_nan,
                ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                output_mem_config));
        } else {
            grad_tensor.emplace_back(ttnn::multiply(grad, inv_scalar, std::nullopt, output_mem_config));
        }
    } else {
        Tensor result = ttnn::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(result);
    }
    return grad_tensor;
}


std::vector<Tensor> ExecuteBackwardDiv::invoke(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    std::string round_mode,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    if (round_mode == "None") {
        Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(other, output_mem_config), std::nullopt, output_mem_config);
        Tensor t_inf = ttnn::operations::creation::full_like(input, std::numeric_limits<float>::infinity(), input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
        Tensor t_nan = ttnn::operations::creation::full_like(input, std::nanf(""), input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(ttnn::where(
            ttnn::eqz(other, output_mem_config),
            ttnn::where(
                ttnn::eqz(grad, output_mem_config),
                t_nan,
                ttnn::multiply(t_inf, ttnn::sign(grad, output_mem_config), std::nullopt, output_mem_config),
                output_mem_config),
            grad_a,
            output_mem_config));
        Tensor grad_b = ttnn::multiply(
            ttnn::neg(grad, output_mem_config),
            (ttnn::multiply(input, ttnn::reciprocal(ttnn::square(other, output_mem_config), output_mem_config), std::nullopt, output_mem_config)),
            std::nullopt,
            output_mem_config);
        grad_tensor.emplace_back(ttnn::where(
            ttnn::eqz(other, output_mem_config),
            ttnn::where(
                ttnn::eqz(grad, output_mem_config),
                t_nan,
                ttnn::where(
                    ttnn::eqz(input, output_mem_config),
                    t_nan,
                    ttnn::multiply(ttnn::multiply(ttnn::neg(t_inf, output_mem_config),
                            ttnn::sign(input, output_mem_config),
                            std::nullopt,
                            output_mem_config),
                        ttnn::sign(grad, output_mem_config),
                        std::nullopt,
                        output_mem_config),
                    output_mem_config),
                output_mem_config),
            grad_b,
            output_mem_config));
    } else {
        Tensor grad_a = ttnn::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(grad_a);
        Tensor grad_b = ttnn::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(grad_b);
    }

    return grad_tensor;
}

std::vector<ComplexTensor> ExecuteBackwardDiv::invoke(
const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor condition_nan = ttnn::logical_and(ttnn::eqz(other.real(),output_mem_config), ttnn::eqz(other.imag(),output_mem_config), std::nullopt, output_mem_config);
    ComplexTensor grad_a = ttnn::operations::complex_binary::_div(grad, ttnn::conj(other,output_mem_config), output_mem_config);
    Tensor grad_a_r = where(condition_nan, ttnn::operations::creation::full_like(grad.real(), std::nanf(""), std::nullopt, std::nullopt, std::nullopt, output_mem_config), ttnn::real(grad_a,output_mem_config),  output_mem_config);
    Tensor grad_a_i = where(condition_nan, ttnn::operations::creation::full_like(grad.imag(), std::nanf(""), std::nullopt, std::nullopt, std::nullopt, output_mem_config), ttnn::imag(grad_a,output_mem_config),  output_mem_config);
    grad_a = ComplexTensor({grad_a_r, grad_a_i});
    grad_a_r.deallocate();
    grad_a_i.deallocate();
    grad_tensor.emplace_back(grad_a);
    ComplexTensor neg_grad = ComplexTensor({ttnn::neg(grad.real(),output_mem_config), ttnn::neg(grad.imag(),output_mem_config)});
    ComplexTensor grad_b = ttnn::operations::complex_binary::_mul(neg_grad, ttnn::conj(ttnn::operations::complex_binary::_div(ttnn::operations::complex_binary::_div(input, other, output_mem_config), other, output_mem_config ),output_mem_config), output_mem_config);
    neg_grad.deallocate();
    Tensor grad_b_r = where(condition_nan, ttnn::operations::creation::full_like(grad.real(), std::nanf(""), std::nullopt, std::nullopt, std::nullopt, output_mem_config), ttnn::real(grad_b,output_mem_config),  output_mem_config);
    Tensor grad_b_i = where(condition_nan, ttnn::operations::creation::full_like(grad.imag(), std::nanf(""), std::nullopt, std::nullopt, std::nullopt, output_mem_config), ttnn::imag(grad_b,output_mem_config),  output_mem_config);
    grad_b = ComplexTensor({grad_b_r, grad_b_i});
    grad_b_r.deallocate();
    grad_b_i.deallocate();
    condition_nan.deallocate();
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<Tensor> ExecuteBackwardMul::invoke(
    const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(grad, scalar, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<ComplexTensor> ExecuteBackwardMul::invoke(
    const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a = ttnn::operations::complex_binary::_mul(grad, ttnn::conj(other,output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    ComplexTensor grad_b = ttnn::operations::complex_binary::_mul(grad, ttnn::conj(input,output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> ExecuteBackwardMul::invoke(
    uint8_t queue_id,
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result;
    if (are_required_outputs.at(0)) {
        if(input_grad.has_value()){
            ttnn::multiply(queue_id, grad, other, std::nullopt, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, input_grad);
        } else {
            input_grad = ttnn::multiply(queue_id, grad, other, std::nullopt, output_mem_config);
        }
        result.emplace_back(input_grad);
    } else {
        result.emplace_back(std::nullopt);
    }
    if (are_required_outputs.at(1)) {
        if(other_grad.has_value()){
            ttnn::multiply(queue_id, grad, input, std::nullopt, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, other_grad);
        } else {
            other_grad = ttnn::multiply(queue_id, grad, input, std::nullopt, output_mem_config);
        }
        result.emplace_back(other_grad);
    } else {
        result.emplace_back(std::nullopt);
    }
    return std::move(result);
}

}  // namespace ttnn::operations::binary_backward
