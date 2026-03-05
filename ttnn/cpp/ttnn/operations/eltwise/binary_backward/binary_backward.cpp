// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/decorators.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"

#include "ttnn/operations/eltwise/unary/unary.hpp"

#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_composite_op.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/creation.hpp"
#include "tools/profiler/op_profiler.hpp"

namespace ttnn::operations::binary_backward {

namespace detail {

// to be used for all binary backward ops to create an empty tensor when there's no preallocated output_tensor
void preallocated_tensors_check(
    std::optional<Tensor>& input_grad,
    std::optional<Tensor>& other_grad,
    const Tensor& input,
    const Tensor& other,
    const std::array<bool, 2>& required_outputs) {
    TT_FATAL(required_outputs[0] || required_outputs[1], "At least one gradient is expected to be calculated.");

    if (required_outputs[0] && !input_grad.has_value()) {
        input_grad = ttnn::empty_like(input);
    }
    if (required_outputs[1] && !other_grad.has_value()) {
        other_grad = ttnn::empty_like(other);
    }
}

std::vector<ttnn::Tensor> atan2_bw_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::SQUARE}, EltwiseUnaryWithParam{UnaryOpType::RECIP}};
    Tensor recip_mul = ttnn::multiply(
        grad,
        ttnn::unary_chain(ttnn::hypot(input, other, output_mem_config), ops_chain, output_mem_config),
        std::nullopt,
        output_mem_config);
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

std::vector<std::optional<ttnn::Tensor>> addalpha_bw_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    float alpha,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt, std::nullopt};

    preallocated_tensors_check(
        input_grad, other_grad, input, other, {are_required_outputs[0], are_required_outputs[1]});

    if (are_required_outputs[0]) {
        ttnn::assign(grad, input_grad.value());
        result[0] = input_grad;
    }
    if (are_required_outputs[1]) {
        ttnn::multiply(grad, alpha, std::nullopt, output_mem_config, other_grad);
        result[1] = other_grad;
    }
    return result;
}

std::vector<std::optional<ttnn::Tensor>> subalpha_bw_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    float alpha,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt, std::nullopt};
    preallocated_tensors_check(
        input_grad, other_grad, input, other, {are_required_outputs[0], are_required_outputs[1]});
    if (are_required_outputs.at(0)) {
        ttnn::assign(grad, input_grad.value());
        result[0] = input_grad;
    }
    if (are_required_outputs.at(1)) {
        ttnn::neg(grad, output_mem_config, other_grad);
        ttnn::multiply(other_grad.value(), alpha, std::nullopt, output_mem_config, other_grad);
        result[1] = other_grad;
    }

    return result;
}

std::vector<std::optional<Tensor>> add_bw_scalar_impl(
    const Tensor& grad,
    const Tensor& input,
    float /*alpha*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result;
    input_grad = input_grad.value_or(ttnn::empty_like(input));
    ttnn::assign(grad, input_grad.value());
    result.emplace_back(input_grad);
    return result;
}

std::vector<std::optional<Tensor>> add_bw_tensor_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt, std::nullopt};
    preallocated_tensors_check(
        input_grad, other_grad, input, other, {are_required_outputs[0], are_required_outputs[1]});
    if (are_required_outputs.at(0)) {
        ttnn::assign(grad, input_grad.value());
        result[0] = input_grad;
    }
    if (are_required_outputs.at(1)) {
        ttnn::multiply(grad, 1.0f, std::nullopt, output_mem_config, other_grad);
        result[1] = other_grad;
    }
    return result;
}

std::vector<ComplexTensor> add_bw_complex_impl(
    const ComplexTensor& grad,
    const ComplexTensor& /*input*/,
    const ComplexTensor& /*other*/,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a = grad;
    grad_tensor.emplace_back(grad_a);
    const Tensor& grad_r = grad.real();
    const Tensor& grad_i = grad.imag();
    ComplexTensor grad_b = ComplexTensor(
        {ttnn::multiply(grad_r, alpha, std::nullopt, output_mem_config),
         ttnn::multiply(grad_i, alpha, std::nullopt, output_mem_config)});
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> sub_bw_scalar_impl(
    const Tensor& grad,
    const Tensor& input,
    float /*alpha*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/,
    const std::optional<Tensor>& input_grad) {
    std::vector<std::optional<Tensor>> result;
    result.emplace_back(
        input_grad.has_value() ? ttnn::assign(grad, input_grad.value()) : ttnn::assign(grad, ttnn::empty_like(input)));
    return result;
}

std::vector<std::optional<Tensor>> sub_bw_tensor_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& input_grad,
    const std::optional<Tensor>& other_grad) {
    return ttnn::subalpha_bw(grad, input, other, 1.0f, are_required_outputs, output_mem_config, input_grad, other_grad);
}

std::vector<ComplexTensor> sub_bw_complex_impl(
    const ComplexTensor& grad,
    const ComplexTensor& /*input*/,
    const ComplexTensor& /*other*/,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    const Tensor& grad_r = grad.real();
    const Tensor& grad_i = grad.imag();
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::NEG}, EltwiseUnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, alpha}};
    ComplexTensor grad_b = ComplexTensor(
        {ttnn::unary_chain(grad_r, ops_chain, output_mem_config),
         ttnn::unary_chain(grad_i, ops_chain, output_mem_config)});
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<ttnn::Tensor> xlogy_bw_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad1_result = ttnn::log(other, true, output_mem_config);
    grad1_result = ttnn::where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config),
            ttnn::le(other, 0.0f, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        0.0f,
        ttnn::where(ttnn::ltz(other, output_mem_config), std::nanf(" "), grad1_result, output_mem_config),
        output_mem_config);
    grad1_result = ttnn::where(
        ttnn::eq(input, std::nanf(" "), std::nullopt, output_mem_config),
        std::nanf(" "),
        grad1_result,
        output_mem_config);
    grad1_result = ttnn::multiply(grad, grad1_result, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad1_result);
    Tensor div_result =
        ttnn::multiply(input, ttnn::reciprocal(other, output_mem_config), std::nullopt, output_mem_config);
    Tensor grad2_result = ttnn::multiply(grad, div_result, std::nullopt, output_mem_config);
    grad2_result = where(
        ttnn::eqz(other, output_mem_config),
        ttnn::multiply(
            ttnn::sign(grad, output_mem_config),
            std::numeric_limits<float>::infinity(),
            std::nullopt,
            output_mem_config),
        grad2_result,
        output_mem_config);
    grad2_result = ttnn::where(
        ttnn::eq(other, std::nanf(" "), std::nullopt, output_mem_config),
        std::nanf(" "),
        grad2_result,
        output_mem_config);
    grad_tensor.emplace_back(grad2_result);
    return grad_tensor;
}

std::vector<ttnn::Tensor> hypot_bw_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    Tensor result_recip = ttnn::reciprocal(ttnn::hypot(input, other, output_memory_config), output_memory_config);
    Tensor grad_a = ttnn::multiply(
        grad,
        ttnn::multiply(input, result_recip, std::nullopt, output_memory_config),
        std::nullopt,
        output_memory_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = ttnn::multiply(
        grad,
        ttnn::multiply(other, result_recip, std::nullopt, output_memory_config),
        std::nullopt,
        output_memory_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

// torch reference
// - name: ldexp(Tensor self, Tensor other) -> Tensor
//   self: grad * 2^other
//   other: grad * self * ln(2) * (2^other)
// # M_LN2 = ln(2)= 0.693147180559945309417
std::vector<ttnn::Tensor> ldexp_bw_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    Tensor tpow_o =
        ttnn::multiply(grad, ttnn::rpow(other, 2.0, output_memory_config), std::nullopt, output_memory_config);
    grad_tensor.emplace_back(tpow_o);
    Tensor result = ttnn::multiply(
        input, ttnn::multiply(tpow_o, M_LN2, std::nullopt, output_memory_config), std::nullopt, output_memory_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

/*
Torch Reference:
name: logaddexp(Tensor self, Tensor other) -> Tensor
self: grad / (1 + exp(other - self)).conj()
other: grad / (1 + exp(self - other)).conj()
*/
std::vector<ttnn::Tensor> logaddexp_bw_impl(
    const Tensor& grad,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(
        !(is_block_float(input_a.dtype()) || is_block_float(grad.dtype()) || is_block_float(other.dtype())),
        "BFLOAT8_B/BFLOAT4_B dtypes are not supported !!");

    std::vector<Tensor> grad_tensor;
    Tensor opexp = ttnn::add(
        ttnn::exp(ttnn::subtract(other, input_a, std::nullopt, output_mem_config), false, output_mem_config),
        1,
        std::nullopt,
        output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(opexp, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    opexp = ttnn::add(
        ttnn::exp(ttnn::subtract(input_a, other, std::nullopt, output_mem_config), false, output_mem_config),
        1,
        std::nullopt,
        output_mem_config);
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

std::vector<ttnn::Tensor> logaddexp2_bw_impl(
    const Tensor& grad,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(
        !(is_block_float(input_a.dtype()) || is_block_float(grad.dtype()) || is_block_float(other.dtype())),
        "BFLOAT8_B/BFLOAT4_B dtypes are not supported !!");

    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input_a.memory_config());
    Tensor oppow = ttnn::add(
        ttnn::rpow(ttnn::subtract(other, input_a, std::nullopt, output_memory_config), 2, output_memory_config),
        1,
        std::nullopt,
        output_memory_config);
    Tensor grad_a =
        ttnn::multiply(grad, ttnn::reciprocal(oppow, output_memory_config), std::nullopt, output_memory_config);
    grad_tensor.emplace_back(grad_a);
    oppow = ttnn::add(
        ttnn::rpow(ttnn::subtract(input_a, other, std::nullopt, output_memory_config), 2, output_memory_config),
        1,
        std::nullopt,
        output_memory_config);
    Tensor grad_b =
        ttnn::multiply(grad, ttnn::reciprocal(oppow, output_memory_config), std::nullopt, output_memory_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<Tensor> remainder_bw_tensor_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor result_div = ttnn::div(input, other, false, "floor", std::nullopt, output_mem_config);
    Tensor grad_b = ttnn::multiply(ttnn::neg(grad), result_div, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<Tensor> remainder_bw_scalar_impl(
    const Tensor& grad,
    const Tensor& /*input*/,
    float /*scalar*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> fmod_bw_tensor_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor result_div = ttnn::div(input, other, false, "trunc", std::nullopt, output_mem_config);
    Tensor grad_b = ttnn::multiply(ttnn::neg(grad), result_div, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<Tensor> fmod_bw_scalar_impl(
    const Tensor& grad,
    const Tensor& /*input*/,
    float /*scalar*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<ttnn::Tensor> squared_difference_bw_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor difference = ttnn::subtract(input, other);
    Tensor grad_a = ttnn::multiply(
        ttnn::multiply(grad, difference, std::nullopt, output_mem_config), 2, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = ttnn::multiply(grad_a, -1, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<std::optional<ttnn::Tensor>> assign_bw_tensor_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& /*output_mem_config*/,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<ttnn::Tensor>> grad_tensor = {std::nullopt, std::nullopt};

    preallocated_tensors_check(
        input_grad, other_grad, input, other, {are_required_outputs[0], are_required_outputs[1]});

    if (are_required_outputs[0]) {
        ttnn::assign(grad, input_grad.value());
        grad_tensor[0] = input_grad;
    }
    if (are_required_outputs[1]) {
        ttnn::assign(grad, other_grad.value());
        grad_tensor[1] = other_grad;
    }
    return grad_tensor;
}

std::vector<std::optional<ttnn::Tensor>> assign_bw_single_impl(
    const Tensor& grad,
    const Tensor& /*input*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<ttnn::Tensor>> grad_tensor = {std::nullopt};
    grad_tensor[0] = input_grad.has_value() ? ttnn::assign(grad, input_grad.value()) : grad;
    return grad_tensor;
}

std::vector<std::optional<Tensor>> concat_bw_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    int dim,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& /*memory_config*/,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> grad_tensor = {std::nullopt, std::nullopt};

    preallocated_tensors_check(
        input_grad, other_grad, input, other, {are_required_outputs[0], are_required_outputs[1]});

    if (are_required_outputs[0]) {
        ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end_index = {
            input.padded_shape()[0], input.padded_shape()[1], input.padded_shape()[2], input.padded_shape()[3]};
        ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
        ttnn::slice(grad, start_index, end_index, step, std::nullopt, input_grad);
        grad_tensor[0] = input_grad;
    }

    if (are_required_outputs[1]) {
        ttnn::SmallVector<uint32_t> start_index_2 = {0, 0, 0, 0};
        if (dim == 0) {
            start_index_2 = {input.padded_shape()[0], 0, 0, 0};
        } else if (dim == 1) {
            start_index_2 = {0, input.padded_shape()[1], 0, 0};
        } else if (dim == 2) {
            start_index_2 = {0, 0, input.padded_shape()[2], 0};
        } else if (dim == 3) {
            start_index_2 = {0, 0, 0, input.padded_shape()[3]};
        }
        ttnn::SmallVector<uint32_t> end_index_2 = {
            grad.padded_shape()[0], grad.padded_shape()[1], grad.padded_shape()[2], grad.padded_shape()[3]};
        ttnn::SmallVector<uint32_t> step_2 = {1, 1, 1, 1};
        ttnn::slice(grad, start_index_2, end_index_2, step_2, std::nullopt, other_grad);
        grad_tensor[1] = other_grad;
    }

    return grad_tensor;
}

std::vector<std::optional<ttnn::Tensor>> rsub_bw_impl(
    const Tensor& grad,
    const Tensor& /*input*/,
    const Tensor& /*other*/,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<ttnn::Tensor>> result = {std::nullopt, std::nullopt};

    preallocated_tensors_check(input_grad, other_grad, grad, grad, {are_required_outputs[0], are_required_outputs[1]});
    if (are_required_outputs.at(0)) {
        ttnn::neg(grad, output_mem_config, input_grad);
        result[0] = input_grad;
    }
    if (are_required_outputs.at(1)) {
        ttnn::assign(grad, other_grad.value());
        result[1] = other_grad;
    }
    return result;
}

std::vector<Tensor> bias_gelu_bw_tensor_impl(
    const Tensor& grad,
    const Tensor& input_a,
    const Tensor& input_b,
    const std::string& approximate,
    const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(
        (approximate == "none" || approximate == "tanh"), "Incorrect approximation type (expected 'none', 'tanh')");
    std::vector<Tensor> grad_tensor;
    Tensor input = ttnn::add(input_a, input_b);
    std::vector<std::optional<Tensor>> gelu_result = ttnn::gelu_bw(grad, input, approximate, output_mem_config);
    if (gelu_result[0].has_value()) {
        grad_tensor.push_back(gelu_result[0].value());
        grad_tensor.push_back(gelu_result[0].value());
    }
    return grad_tensor;
}

std::vector<Tensor> bias_gelu_bw_scalar_impl(
    const Tensor& grad,
    const Tensor& input_tensor,
    float bias,
    const std::string& approximate,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    TT_FATAL(
        (approximate == "none" || approximate == "tanh"),
        "Incorrect rounding mode (expected 'none' or 'tanh')",
        "Error");
    Tensor input = ttnn::add(input_tensor, bias);
    std::vector<std::optional<Tensor>> gelu_result = ttnn::gelu_bw(grad, input, approximate, output_mem_config);
    if (gelu_result[0].has_value()) {
        grad_tensor.push_back(gelu_result[0].value());
    }
    return grad_tensor;
}

}  // namespace detail

// template parameter min_or_max = TRUE for MAX, FALSE for MIN
template <bool min_or_max>
std::vector<Tensor> _min_or_max_bw(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_scale_grad = ttnn::multiply(grad, 0.5, std::nullopt, output_mem_config);
    Tensor t_sub = ttnn::subtract(other, input, std::nullopt, output_mem_config);
    Tensor t_sub_gtz = ttnn::gtz(t_sub, output_mem_config);
    Tensor t_sub_eqz = ttnn::eqz(t_sub, output_mem_config);
    Tensor t_sub_ltz = ttnn::ltz(t_sub, output_mem_config);
    Tensor grad_other = ttnn::add(
        ttnn::multiply(t_sub_ltz, grad, std::nullopt, output_mem_config),
        ttnn::multiply(t_sub_eqz, t_scale_grad, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    Tensor grad_input = ttnn::add(
        ttnn::multiply(t_sub_gtz, grad, std::nullopt, output_mem_config),
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

std::vector<ttnn::Tensor> max_bw_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return _min_or_max_bw<true>(grad, input, other, output_mem_config);
}

std::vector<ttnn::Tensor> min_bw_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return _min_or_max_bw<false>(grad, input, other, output_mem_config);
}

std::vector<std::optional<ttnn::Tensor>> div_bw_scalar_impl(
    const Tensor& grad,
    const Tensor& input,
    float scalar,
    const std::optional<std::string>& rounding_mode,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    TT_FATAL(
        (rounding_mode == std::nullopt || rounding_mode == "trunc" || rounding_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");

    std::vector<std::optional<Tensor>> result;
    input_grad = input_grad.value_or(ttnn::empty_like(input));

    if (rounding_mode == std::nullopt) {
        float t_inf = std::numeric_limits<float>::infinity();
        if (scalar == 0.0) {
            float t_nan = std::nanf("");
            where(
                ttnn::eqz(grad, output_mem_config),
                t_nan,
                ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                output_mem_config,
                input_grad);
            result.push_back(input_grad);
        } else {
            float inv_scalar = 1.0f / scalar;
            ttnn::multiply(grad, inv_scalar, std::nullopt, output_mem_config, input_grad);
            result.push_back(input_grad);
        }
    } else {
        ttnn::zeros_like(grad, grad.dtype(), grad.layout(), std::nullopt, output_mem_config, input_grad);
        result.push_back(input_grad);
    }
    return result;
}

std::vector<std::optional<ttnn::Tensor>> div_bw_tensor_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<std::string>& rounding_mode,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt, std::nullopt};
    detail::preallocated_tensors_check(
        input_grad, other_grad, input, other, {are_required_outputs[0], are_required_outputs[1]});
    TT_FATAL(
        (rounding_mode == std::nullopt || rounding_mode == "trunc" || rounding_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");

    if (rounding_mode == std::nullopt) {
        float t_nan = std::nanf("");
        float t_inf = std::numeric_limits<float>::infinity();
        float neg_inf = -std::numeric_limits<float>::infinity();
        if (are_required_outputs.at(0)) {
            ttnn::multiply(
                grad, ttnn::reciprocal(other, output_mem_config), std::nullopt, output_mem_config, input_grad);
            ttnn::where(
                ttnn::eqz(other, output_mem_config),
                ttnn::where(
                    ttnn::eqz(grad, output_mem_config),
                    t_nan,
                    ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                    output_mem_config),
                input_grad.value(),
                output_mem_config);
            result[0] = input_grad;
        }
        if (are_required_outputs.at(1)) {
            ttnn::multiply(
                ttnn::neg(grad, output_mem_config),
                (ttnn::multiply(
                    input,
                    ttnn::reciprocal(ttnn::square(other, output_mem_config), output_mem_config),
                    std::nullopt,
                    output_mem_config)),
                std::nullopt,
                output_mem_config,
                other_grad);
            ttnn::where(
                ttnn::eqz(other, output_mem_config),
                ttnn::where(
                    ttnn::eqz(grad, output_mem_config),
                    t_nan,
                    ttnn::where(
                        ttnn::eqz(input, output_mem_config),
                        t_nan,
                        ttnn::multiply(
                            ttnn::multiply(
                                ttnn::sign(input, output_mem_config), neg_inf, std::nullopt, output_mem_config),
                            ttnn::sign(grad, output_mem_config),
                            std::nullopt,
                            output_mem_config),
                        output_mem_config),
                    output_mem_config),
                other_grad.value(),
                output_mem_config);
            result[1] = other_grad;
        }
    } else {
        if (are_required_outputs.at(0)) {
            ttnn::zeros_like(grad, grad.dtype(), grad.layout(), std::nullopt, output_mem_config, input_grad);
            result[0] = input_grad;
        }
        if (are_required_outputs.at(1)) {
            ttnn::zeros_like(grad, grad.dtype(), grad.layout(), std::nullopt, output_mem_config, other_grad);
            result[1] = other_grad;
        }
    }
    return result;
}

std::vector<ComplexTensor> div_bw_complex_impl(
    const ComplexTensor& grad,
    const ComplexTensor& input,
    const ComplexTensor& other,
    const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor condition_nan = ttnn::logical_and(
        ttnn::eqz(other.real(), output_mem_config),
        ttnn::eqz(other.imag(), output_mem_config),
        std::nullopt,
        output_mem_config);
    ComplexTensor grad_a =
        ttnn::operations::complex_binary::_div(grad, ttnn::conj(other, output_mem_config), output_mem_config);
    Tensor grad_a_r = where(condition_nan, std::nanf(""), ttnn::real(grad_a, output_mem_config), output_mem_config);
    Tensor grad_a_i = where(condition_nan, std::nanf(""), ttnn::imag(grad_a, output_mem_config), output_mem_config);
    grad_a = ComplexTensor({grad_a_r, grad_a_i});
    grad_a_r.deallocate();
    grad_a_i.deallocate();
    grad_tensor.emplace_back(grad_a);
    ComplexTensor neg_grad =
        ComplexTensor({ttnn::neg(grad.real(), output_mem_config), ttnn::neg(grad.imag(), output_mem_config)});
    ComplexTensor grad_b = ttnn::operations::complex_binary::_mul(
        neg_grad,
        ttnn::conj(
            ttnn::operations::complex_binary::_div(
                ttnn::operations::complex_binary::_div(input, other, output_mem_config), other, output_mem_config),
            output_mem_config),
        output_mem_config);
    neg_grad.deallocate();
    Tensor grad_b_r = where(condition_nan, std::nanf(""), ttnn::real(grad_b, output_mem_config), output_mem_config);
    Tensor grad_b_i = where(condition_nan, std::nanf(""), ttnn::imag(grad_b, output_mem_config), output_mem_config);
    grad_b = ComplexTensor({grad_b_r, grad_b_i});
    grad_b_r.deallocate();
    grad_b_i.deallocate();
    condition_nan.deallocate();
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<std::optional<ttnn::Tensor>> mul_bw_scalar_impl(
    const Tensor& grad,
    const Tensor& /*input*/,
    float scalar,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result;
    if (!input_grad.has_value()) {
        input_grad = ttnn::empty_like(grad);
    }
    ttnn::multiply(grad, scalar, std::nullopt, output_mem_config, input_grad);
    result.push_back(input_grad);
    return result;
}

std::vector<ComplexTensor> mul_bw_complex_impl(
    const ComplexTensor& grad,
    const ComplexTensor& input,
    const ComplexTensor& other,
    const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a =
        ttnn::operations::complex_binary::_mul(grad, ttnn::conj(other, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    ComplexTensor grad_b =
        ttnn::operations::complex_binary::_mul(grad, ttnn::conj(input, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> mul_bw_tensor_impl(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt, std::nullopt};
    detail::preallocated_tensors_check(
        input_grad, other_grad, input, other, {are_required_outputs[0], are_required_outputs[1]});

    if (are_required_outputs.at(0)) {
        ttnn::multiply(grad, other, std::nullopt, output_mem_config, input_grad);
        result[0] = input_grad;
    }
    if (are_required_outputs.at(1)) {
        ttnn::multiply(grad, input, std::nullopt, output_mem_config, other_grad);
        result[1] = other_grad;
    }
    return result;
}

}  // namespace ttnn::operations::binary_backward

namespace ttnn {

std::vector<Tensor> atan2_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::atan2_bw_impl(grad_tensor, input_a, other, output_mem_config);
}

std::vector<Tensor> xlogy_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::xlogy_bw_impl(grad_tensor, input_a, other, output_mem_config);
}

std::vector<Tensor> hypot_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::hypot_bw_impl(grad_tensor, input_a, other, output_mem_config);
}

std::vector<Tensor> ldexp_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::ldexp_bw_impl(grad_tensor, input_a, other, output_mem_config);
}

std::vector<Tensor> logaddexp_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::logaddexp_bw_impl(grad_tensor, input_a, other, output_mem_config);
}

std::vector<Tensor> logaddexp2_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::logaddexp2_bw_impl(grad_tensor, input_a, other, output_mem_config);
}

std::vector<Tensor> squared_difference_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::squared_difference_bw_impl(
        grad_tensor, input_a, other, output_mem_config);
}

std::vector<Tensor> min_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::min_bw_impl(grad_tensor, input_a, other, output_mem_config);
}

std::vector<Tensor> max_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::max_bw_impl(grad_tensor, input_a, other, output_mem_config);
}

std::vector<std::optional<Tensor>> subalpha_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    float alpha,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    return operations::binary_backward::detail::subalpha_bw_impl(
        grad_tensor, input_a, other, alpha, are_required_outputs, output_mem_config, input_grad, other_grad);
}

std::vector<std::optional<Tensor>> rsub_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    return operations::binary_backward::detail::rsub_bw_impl(
        grad_tensor, input_a, other, are_required_outputs, output_mem_config, input_grad, other_grad);
}

std::vector<std::optional<Tensor>> concat_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_a_arg,
    const Tensor& other,
    int dim,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    return operations::binary_backward::detail::concat_bw_impl(
        grad_tensor_arg, input_tensor_a_arg, other, dim, are_required_outputs, memory_config, input_grad, other_grad);
}

std::vector<std::optional<Tensor>> mul_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float scalar,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    return operations::binary_backward::mul_bw_scalar_impl(
        grad_tensor_arg, input_tensor_arg, scalar, output_mem_config, input_grad);
}

std::vector<std::optional<Tensor>> mul_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const Tensor& other_tensor_arg,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    return operations::binary_backward::mul_bw_tensor_impl(
        grad_tensor_arg,
        input_tensor_arg,
        other_tensor_arg,
        are_required_outputs,
        output_mem_config,
        input_grad,
        other_grad);
}

std::vector<ComplexTensor> mul_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& input_a,
    const ComplexTensor& other,
    const MemoryConfig& output_mem_config) {
    return operations::binary_backward::mul_bw_complex_impl(grad_tensor, input_a, other, output_mem_config);
}

std::vector<std::optional<Tensor>> assign_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    return operations::binary_backward::detail::assign_bw_single_impl(
        grad_tensor, input_tensor, output_mem_config, input_grad);
}

std::vector<std::optional<Tensor>> assign_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    const Tensor& other_tensor,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    return operations::binary_backward::detail::assign_bw_tensor_impl(
        grad_tensor, input_tensor, other_tensor, are_required_outputs, output_mem_config, input_grad, other_grad);
}

std::vector<Tensor> bias_gelu_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& input_b,
    const std::string& approximate,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::bias_gelu_bw_tensor_impl(
        grad_tensor, input_a, input_b, approximate, output_mem_config);
}

std::vector<Tensor> bias_gelu_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float bias,
    const std::string& approximate,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::bias_gelu_bw_scalar_impl(
        grad_tensor, input_tensor, bias, approximate, output_mem_config);
}

std::vector<std::optional<Tensor>> addalpha_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    float alpha,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    return operations::binary_backward::detail::addalpha_bw_impl(
        grad_tensor, input_a, other, alpha, are_required_outputs, output_mem_config, input_grad, other_grad);
}

std::vector<std::optional<Tensor>> add_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    return operations::binary_backward::detail::add_bw_scalar_impl(
        grad_tensor, input_tensor, alpha, output_mem_config, input_grad);
}

std::vector<std::optional<Tensor>> add_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    return operations::binary_backward::detail::add_bw_tensor_impl(
        grad_tensor, input_a, other, are_required_outputs, output_mem_config, input_grad, other_grad);
}

std::vector<ComplexTensor> add_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& input_a,
    const ComplexTensor& other,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::add_bw_complex_impl(
        grad_tensor, input_a, other, alpha, output_mem_config);
}

std::vector<std::optional<Tensor>> sub_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& input_grad) {
    return operations::binary_backward::detail::sub_bw_scalar_impl(
        grad_tensor, input_tensor, alpha, output_mem_config, input_grad);
}

std::vector<std::optional<Tensor>> sub_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& input_grad,
    const std::optional<Tensor>& other_grad) {
    return operations::binary_backward::detail::sub_bw_tensor_impl(
        grad_tensor, input_a, other, are_required_outputs, output_mem_config, input_grad, other_grad);
}

std::vector<ComplexTensor> sub_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& input_a,
    const ComplexTensor& other,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::sub_bw_complex_impl(
        grad_tensor, input_a, other, alpha, output_mem_config);
}

std::vector<std::optional<Tensor>> div_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float scalar,
    const std::optional<std::string>& rounding_mode,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    return operations::binary_backward::div_bw_scalar_impl(
        grad_tensor, input_tensor, scalar, rounding_mode, output_mem_config, input_grad);
}

std::vector<std::optional<Tensor>> div_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<std::string>& rounding_mode,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    return operations::binary_backward::div_bw_tensor_impl(
        grad_tensor, input_a, other, rounding_mode, are_required_outputs, output_mem_config, input_grad, other_grad);
}

std::vector<ComplexTensor> div_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& input_a,
    const ComplexTensor& other,
    const MemoryConfig& output_mem_config) {
    return operations::binary_backward::div_bw_complex_impl(grad_tensor, input_a, other, output_mem_config);
}

std::vector<Tensor> remainder_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float scalar,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::remainder_bw_scalar_impl(
        grad_tensor, input_tensor, scalar, output_mem_config);
}

std::vector<Tensor> remainder_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::remainder_bw_tensor_impl(
        grad_tensor, input_a, other, output_mem_config);
}

std::vector<Tensor> fmod_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float scalar,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::fmod_bw_scalar_impl(
        grad_tensor, input_tensor, scalar, output_mem_config);
}

std::vector<Tensor> fmod_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::detail::fmod_bw_tensor_impl(grad_tensor, input_a, other, output_mem_config);
}

}  // namespace ttnn
