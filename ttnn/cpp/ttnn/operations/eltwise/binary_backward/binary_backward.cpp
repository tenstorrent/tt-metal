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
#include "ttnn/operations/creation/creation.hpp"
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

}  // namespace ttnn::operations::binary_backward

namespace ttnn {

std::vector<Tensor> atan2_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad;
    float t_nan = std::nanf("");
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::SQUARE}, EltwiseUnaryWithParam{UnaryOpType::RECIP}};
    Tensor recip_mul = ttnn::multiply(
        grad_tensor,
        ttnn::unary_chain(ttnn::hypot(input_a, other, output_mem_config), ops_chain, output_mem_config),
        std::nullopt,
        output_mem_config);
    Tensor grad_a = ttnn::multiply(other, recip_mul, std::nullopt, output_mem_config);
    Tensor cond = ttnn::logical_and(ttnn::eqz(input_a, output_mem_config), ttnn::eqz(other, output_mem_config));
    grad_a = ttnn::where(cond, t_nan, grad_a, output_mem_config);
    grad.emplace_back(grad_a);
    Tensor grad_b = ttnn::multiply(ttnn::neg(input_a), recip_mul, std::nullopt, output_mem_config);
    grad_b = ttnn::where(cond, t_nan, grad_b, output_mem_config);
    recip_mul.deallocate();
    cond.deallocate();
    grad.emplace_back(grad_b);
    return grad;
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
    std::vector<std::optional<Tensor>> result = {std::nullopt, std::nullopt};

    operations::binary_backward::detail::preallocated_tensors_check(
        input_grad, other_grad, input_a, other, {are_required_outputs[0], are_required_outputs[1]});

    if (are_required_outputs[0]) {
        ttnn::assign(grad_tensor, input_grad.value());
        result[0] = input_grad;
    }
    if (are_required_outputs[1]) {
        ttnn::multiply(grad_tensor, alpha, std::nullopt, output_mem_config, other_grad);
        result[1] = other_grad;
    }
    return result;
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
    std::vector<std::optional<Tensor>> result = {std::nullopt, std::nullopt};
    operations::binary_backward::detail::preallocated_tensors_check(
        input_grad, other_grad, input_a, other, {are_required_outputs[0], are_required_outputs[1]});
    if (are_required_outputs.at(0)) {
        ttnn::assign(grad_tensor, input_grad.value());
        result[0] = input_grad;
    }
    if (are_required_outputs.at(1)) {
        ttnn::neg(grad_tensor, output_mem_config, other_grad);
        ttnn::multiply(other_grad.value(), alpha, std::nullopt, output_mem_config, other_grad);
        result[1] = other_grad;
    }

    return result;
}

std::vector<std::optional<Tensor>> add_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float /*alpha*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result;
    input_grad = input_grad.value_or(ttnn::empty_like(input_tensor));
    ttnn::assign(grad_tensor, input_grad.value());
    result.emplace_back(input_grad);
    return result;
}

std::vector<std::optional<Tensor>> add_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt, std::nullopt};
    operations::binary_backward::detail::preallocated_tensors_check(
        input_grad, other_grad, input_a, other, {are_required_outputs[0], are_required_outputs[1]});
    if (are_required_outputs.at(0)) {
        ttnn::assign(grad_tensor, input_grad.value());
        result[0] = input_grad;
    }
    if (are_required_outputs.at(1)) {
        ttnn::multiply(grad_tensor, 1.0f, std::nullopt, output_mem_config, other_grad);
        result[1] = other_grad;
    }
    return result;
}

std::vector<ComplexTensor> add_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& /*input_a*/,
    const ComplexTensor& /*other*/,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor_res;
    ComplexTensor grad_a = grad_tensor;
    grad_tensor_res.emplace_back(grad_a);
    const Tensor& grad_r = grad_tensor.real();
    const Tensor& grad_i = grad_tensor.imag();
    ComplexTensor grad_b = ComplexTensor(
        {ttnn::multiply(grad_r, alpha, std::nullopt, output_mem_config),
         ttnn::multiply(grad_i, alpha, std::nullopt, output_mem_config)});
    grad_tensor_res.emplace_back(grad_b);
    return grad_tensor_res;
}

std::vector<std::optional<Tensor>> sub_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float /*alpha*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result;
    result.emplace_back(
        input_grad.has_value() ? ttnn::assign(grad_tensor, input_grad.value())
                               : ttnn::assign(grad_tensor, ttnn::empty_like(input_tensor)));
    return result;
}

std::vector<std::optional<Tensor>> sub_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& input_grad,
    const std::optional<Tensor>& other_grad) {
    return ttnn::subalpha_bw(
        grad_tensor, input_a, other, 1.0f, are_required_outputs, output_mem_config, input_grad, other_grad);
}

std::vector<ComplexTensor> sub_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& /*input_a*/,
    const ComplexTensor& /*other*/,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor_res;
    grad_tensor_res.emplace_back(grad_tensor);
    const Tensor& grad_r = grad_tensor.real();
    const Tensor& grad_i = grad_tensor.imag();
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::NEG}, EltwiseUnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, alpha}};
    ComplexTensor grad_b = ComplexTensor(
        {ttnn::unary_chain(grad_r, ops_chain, output_mem_config),
         ttnn::unary_chain(grad_i, ops_chain, output_mem_config)});
    grad_tensor_res.emplace_back(grad_b);
    return grad_tensor_res;
}

std::vector<Tensor> xlogy_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad;
    Tensor grad1_result = ttnn::log(other, true, output_mem_config);
    grad1_result = ttnn::where(
        ttnn::logical_and(
            ttnn::eqz(input_a, output_mem_config),
            ttnn::le(other, 0.0f, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        0.0f,
        ttnn::where(ttnn::ltz(other, output_mem_config), std::nanf(" "), grad1_result, output_mem_config),
        output_mem_config);
    grad1_result = ttnn::where(
        ttnn::eq(input_a, std::nanf(" "), std::nullopt, output_mem_config),
        std::nanf(" "),
        grad1_result,
        output_mem_config);
    grad1_result = ttnn::multiply(grad_tensor, grad1_result, std::nullopt, output_mem_config);

    grad.emplace_back(grad1_result);
    Tensor div_result =
        ttnn::multiply(input_a, ttnn::reciprocal(other, output_mem_config), std::nullopt, output_mem_config);
    Tensor grad2_result = ttnn::multiply(grad_tensor, div_result, std::nullopt, output_mem_config);
    grad2_result = where(
        ttnn::eqz(other, output_mem_config),
        ttnn::multiply(
            ttnn::sign(grad_tensor, output_mem_config),
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
    grad.emplace_back(grad2_result);
    return grad;
}

std::vector<Tensor> hypot_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad;
    auto output_memory_config = output_mem_config.value_or(input_a.memory_config());
    Tensor result_recip = ttnn::reciprocal(ttnn::hypot(input_a, other, output_memory_config), output_memory_config);
    Tensor grad_a = ttnn::multiply(
        grad_tensor,
        ttnn::multiply(input_a, result_recip, std::nullopt, output_memory_config),
        std::nullopt,
        output_memory_config);
    grad.emplace_back(grad_a);
    Tensor grad_b = ttnn::multiply(
        grad_tensor,
        ttnn::multiply(other, result_recip, std::nullopt, output_memory_config),
        std::nullopt,
        output_memory_config);
    grad.emplace_back(grad_b);
    return grad;
}

std::vector<Tensor> ldexp_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad;
    auto output_memory_config = output_mem_config.value_or(input_a.memory_config());
    Tensor tpow_o =
        ttnn::multiply(grad_tensor, ttnn::rpow(other, 2.0, output_memory_config), std::nullopt, output_memory_config);
    grad.emplace_back(tpow_o);
    Tensor result = ttnn::multiply(
        input_a, ttnn::multiply(tpow_o, M_LN2, std::nullopt, output_memory_config), std::nullopt, output_memory_config);
    grad.emplace_back(result);
    return grad;
}

std::vector<Tensor> logaddexp_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(
        !(tt::tt_metal::is_block_float(input_a.dtype()) || tt::tt_metal::is_block_float(grad_tensor.dtype()) ||
          tt::tt_metal::is_block_float(other.dtype())),
        "BFLOAT8_B/BFLOAT4_B dtypes are not supported !!");

    std::vector<Tensor> grad;
    Tensor opexp = ttnn::add(
        ttnn::exp(ttnn::subtract(other, input_a, std::nullopt, output_mem_config), false, output_mem_config),
        1,
        std::nullopt,
        output_mem_config);
    Tensor grad_a =
        ttnn::multiply(grad_tensor, ttnn::reciprocal(opexp, output_mem_config), std::nullopt, output_mem_config);
    grad.emplace_back(grad_a);
    opexp = ttnn::add(
        ttnn::exp(ttnn::subtract(input_a, other, std::nullopt, output_mem_config), false, output_mem_config),
        1,
        std::nullopt,
        output_mem_config);
    Tensor grad_b =
        ttnn::multiply(grad_tensor, ttnn::reciprocal(opexp, output_mem_config), std::nullopt, output_mem_config);
    grad.emplace_back(grad_b);
    return grad;
}

std::vector<Tensor> logaddexp2_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(
        !(tt::tt_metal::is_block_float(input_a.dtype()) || tt::tt_metal::is_block_float(grad_tensor.dtype()) ||
          tt::tt_metal::is_block_float(other.dtype())),
        "BFLOAT8_B/BFLOAT4_B dtypes are not supported !!");

    std::vector<Tensor> grad;
    auto output_memory_config = output_mem_config.value_or(input_a.memory_config());
    Tensor oppow = ttnn::add(
        ttnn::rpow(ttnn::subtract(other, input_a, std::nullopt, output_memory_config), 2, output_memory_config),
        1,
        std::nullopt,
        output_memory_config);
    Tensor grad_a =
        ttnn::multiply(grad_tensor, ttnn::reciprocal(oppow, output_memory_config), std::nullopt, output_memory_config);
    grad.emplace_back(grad_a);
    oppow = ttnn::add(
        ttnn::rpow(ttnn::subtract(input_a, other, std::nullopt, output_memory_config), 2, output_memory_config),
        1,
        std::nullopt,
        output_memory_config);
    Tensor grad_b =
        ttnn::multiply(grad_tensor, ttnn::reciprocal(oppow, output_memory_config), std::nullopt, output_memory_config);
    grad.emplace_back(grad_b);
    return grad;
}

std::vector<Tensor> remainder_bw(
    const Tensor& grad_tensor,
    const Tensor& /*input_tensor*/,
    float /*scalar*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad;
    grad.emplace_back(grad_tensor);
    return grad;
}

std::vector<Tensor> remainder_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad;
    grad.emplace_back(grad_tensor);
    Tensor result_div = ttnn::div(input_a, other, false, "floor", std::nullopt, output_mem_config);
    Tensor grad_b = ttnn::multiply(ttnn::neg(grad_tensor), result_div, std::nullopt, output_mem_config);
    grad.emplace_back(grad_b);
    return grad;
}

std::vector<Tensor> fmod_bw(
    const Tensor& grad_tensor,
    const Tensor& /*input_tensor*/,
    float /*scalar*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad;
    grad.emplace_back(grad_tensor);
    return grad;
}

std::vector<Tensor> fmod_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad;
    grad.emplace_back(grad_tensor);
    Tensor result_div = ttnn::div(input_a, other, false, "trunc", std::nullopt, output_mem_config);
    Tensor grad_b = ttnn::multiply(ttnn::neg(grad_tensor), result_div, std::nullopt, output_mem_config);
    grad.emplace_back(grad_b);
    return grad;
}

std::vector<Tensor> squared_difference_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad;
    Tensor difference = ttnn::subtract(input_a, other);
    Tensor grad_a = ttnn::multiply(
        ttnn::multiply(grad_tensor, difference, std::nullopt, output_mem_config), 2, std::nullopt, output_mem_config);
    grad.emplace_back(grad_a);
    Tensor grad_b = ttnn::multiply(grad_a, -1, std::nullopt, output_mem_config);
    grad.emplace_back(grad_b);
    return grad;
}

std::vector<std::optional<Tensor>> assign_bw(
    const Tensor& grad_tensor,
    const Tensor& /*input_tensor*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<ttnn::Tensor>> grad_tensor_res = {std::nullopt};
    grad_tensor_res[0] = input_grad.has_value() ? ttnn::assign(grad_tensor, input_grad.value()) : grad_tensor;
    return grad_tensor_res;
}

std::vector<std::optional<Tensor>> assign_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    const Tensor& other_tensor,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& /*output_mem_config*/,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<ttnn::Tensor>> grad_tensor_res = {std::nullopt, std::nullopt};

    operations::binary_backward::detail::preallocated_tensors_check(
        input_grad, other_grad, input_tensor, other_tensor, {are_required_outputs[0], are_required_outputs[1]});

    if (are_required_outputs[0]) {
        ttnn::assign(grad_tensor, input_grad.value());
        grad_tensor_res[0] = input_grad;
    }
    if (are_required_outputs[1]) {
        ttnn::assign(grad_tensor, other_grad.value());
        grad_tensor_res[1] = other_grad;
    }
    return grad_tensor_res;
}

std::vector<std::optional<Tensor>> concat_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_a_arg,
    const Tensor& other,
    int dim,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& /*memory_config*/,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> grad_tensor = {std::nullopt, std::nullopt};

    operations::binary_backward::detail::preallocated_tensors_check(
        input_grad, other_grad, input_tensor_a_arg, other, {are_required_outputs[0], are_required_outputs[1]});

    if (are_required_outputs[0]) {
        ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end_index = {
            input_tensor_a_arg.padded_shape()[0],
            input_tensor_a_arg.padded_shape()[1],
            input_tensor_a_arg.padded_shape()[2],
            input_tensor_a_arg.padded_shape()[3]};
        ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
        ttnn::slice(grad_tensor_arg, start_index, end_index, step, std::nullopt, input_grad);
        grad_tensor[0] = input_grad;
    }

    if (are_required_outputs[1]) {
        ttnn::SmallVector<uint32_t> start_index_2 = {0, 0, 0, 0};
        if (dim == 0) {
            start_index_2 = {input_tensor_a_arg.padded_shape()[0], 0, 0, 0};
        } else if (dim == 1) {
            start_index_2 = {0, input_tensor_a_arg.padded_shape()[1], 0, 0};
        } else if (dim == 2) {
            start_index_2 = {0, 0, input_tensor_a_arg.padded_shape()[2], 0};
        } else if (dim == 3) {
            start_index_2 = {0, 0, 0, input_tensor_a_arg.padded_shape()[3]};
        }
        ttnn::SmallVector<uint32_t> end_index_2 = {
            grad_tensor_arg.padded_shape()[0],
            grad_tensor_arg.padded_shape()[1],
            grad_tensor_arg.padded_shape()[2],
            grad_tensor_arg.padded_shape()[3]};
        ttnn::SmallVector<uint32_t> step_2 = {1, 1, 1, 1};
        ttnn::slice(grad_tensor_arg, start_index_2, end_index_2, step_2, std::nullopt, other_grad);
        grad_tensor[1] = other_grad;
    }

    return grad_tensor;
}

std::vector<std::optional<Tensor>> rsub_bw(
    const Tensor& grad_tensor,
    const Tensor& /*input_a*/,
    const Tensor& /*other*/,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<ttnn::Tensor>> result = {std::nullopt, std::nullopt};

    operations::binary_backward::detail::preallocated_tensors_check(
        input_grad, other_grad, grad_tensor, grad_tensor, {are_required_outputs[0], are_required_outputs[1]});
    if (are_required_outputs.at(0)) {
        ttnn::neg(grad_tensor, output_mem_config, input_grad);
        result[0] = input_grad;
    }
    if (are_required_outputs.at(1)) {
        ttnn::assign(grad_tensor, other_grad.value());
        result[1] = other_grad;
    }
    return result;
}

std::vector<Tensor> bias_gelu_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& input_b,
    const std::string& approximate,
    const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(
        (approximate == "none" || approximate == "tanh"), "Incorrect approximation type (expected 'none', 'tanh')");
    std::vector<Tensor> grad_tensor_res;
    Tensor input = ttnn::add(input_a, input_b);
    std::vector<std::optional<Tensor>> gelu_result = ttnn::gelu_bw(grad_tensor, input, approximate, output_mem_config);
    if (gelu_result[0].has_value()) {
        grad_tensor_res.push_back(gelu_result[0].value());
        grad_tensor_res.push_back(gelu_result[0].value());
    }
    return grad_tensor_res;
}

std::vector<Tensor> bias_gelu_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float bias,
    const std::string& approximate,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor_res;
    TT_FATAL(
        (approximate == "none" || approximate == "tanh"),
        "Incorrect rounding mode (expected 'none' or 'tanh')",
        "Error");
    Tensor input = ttnn::add(input_tensor, bias);
    std::vector<std::optional<Tensor>> gelu_result = ttnn::gelu_bw(grad_tensor, input, approximate, output_mem_config);
    if (gelu_result[0].has_value()) {
        grad_tensor_res.push_back(gelu_result[0].value());
    }
    return grad_tensor_res;
}

std::vector<Tensor> max_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::_min_or_max_bw<true>(grad_tensor, input_a, other, output_mem_config);
}

std::vector<Tensor> min_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config) {
    return operations::binary_backward::_min_or_max_bw<false>(grad_tensor, input_a, other, output_mem_config);
}

std::vector<std::optional<Tensor>> div_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float scalar,
    const std::optional<std::string>& rounding_mode,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    TT_FATAL(
        (rounding_mode == std::nullopt || rounding_mode == "trunc" || rounding_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");

    std::vector<std::optional<Tensor>> result;
    input_grad = input_grad.value_or(ttnn::empty_like(input_tensor));

    if (rounding_mode == std::nullopt) {
        float t_inf = std::numeric_limits<float>::infinity();
        if (scalar == 0.0) {
            float t_nan = std::nanf("");
            where(
                ttnn::eqz(grad_tensor, output_mem_config),
                t_nan,
                ttnn::multiply(ttnn::sign(grad_tensor, output_mem_config), t_inf, std::nullopt, output_mem_config),
                output_mem_config,
                input_grad);
            result.push_back(input_grad);
        } else {
            float inv_scalar = 1.0f / scalar;
            ttnn::multiply(grad_tensor, inv_scalar, std::nullopt, output_mem_config, input_grad);
            result.push_back(input_grad);
        }
    } else {
        ttnn::zeros_like(
            grad_tensor, grad_tensor.dtype(), grad_tensor.layout(), std::nullopt, output_mem_config, input_grad);
        result.push_back(input_grad);
    }
    return result;
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
    std::vector<std::optional<Tensor>> result = {std::nullopt, std::nullopt};
    operations::binary_backward::detail::preallocated_tensors_check(
        input_grad, other_grad, input_a, other, {are_required_outputs[0], are_required_outputs[1]});
    TT_FATAL(
        (rounding_mode == std::nullopt || rounding_mode == "trunc" || rounding_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");

    if (rounding_mode == std::nullopt) {
        float t_nan = std::nanf("");
        float t_inf = std::numeric_limits<float>::infinity();
        float neg_inf = -std::numeric_limits<float>::infinity();
        if (are_required_outputs.at(0)) {
            ttnn::multiply(
                grad_tensor, ttnn::reciprocal(other, output_mem_config), std::nullopt, output_mem_config, input_grad);
            ttnn::where(
                ttnn::eqz(other, output_mem_config),
                ttnn::where(
                    ttnn::eqz(grad_tensor, output_mem_config),
                    t_nan,
                    ttnn::multiply(ttnn::sign(grad_tensor, output_mem_config), t_inf, std::nullopt, output_mem_config),
                    output_mem_config),
                input_grad.value(),
                output_mem_config);
            result[0] = input_grad;
        }
        if (are_required_outputs.at(1)) {
            ttnn::multiply(
                ttnn::neg(grad_tensor, output_mem_config),
                (ttnn::multiply(
                    input_a,
                    ttnn::reciprocal(ttnn::square(other, output_mem_config), output_mem_config),
                    std::nullopt,
                    output_mem_config)),
                std::nullopt,
                output_mem_config,
                other_grad);
            ttnn::where(
                ttnn::eqz(other, output_mem_config),
                ttnn::where(
                    ttnn::eqz(grad_tensor, output_mem_config),
                    t_nan,
                    ttnn::where(
                        ttnn::eqz(input_a, output_mem_config),
                        t_nan,
                        ttnn::multiply(
                            ttnn::multiply(
                                ttnn::sign(input_a, output_mem_config), neg_inf, std::nullopt, output_mem_config),
                            ttnn::sign(grad_tensor, output_mem_config),
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
            ttnn::zeros_like(
                grad_tensor, grad_tensor.dtype(), grad_tensor.layout(), std::nullopt, output_mem_config, input_grad);
            result[0] = input_grad;
        }
        if (are_required_outputs.at(1)) {
            ttnn::zeros_like(
                grad_tensor, grad_tensor.dtype(), grad_tensor.layout(), std::nullopt, output_mem_config, other_grad);
            result[1] = other_grad;
        }
    }
    return result;
}

std::vector<ComplexTensor> div_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& input_a,
    const ComplexTensor& other,
    const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor_res;
    Tensor condition_nan = ttnn::logical_and(
        ttnn::eqz(other.real(), output_mem_config),
        ttnn::eqz(other.imag(), output_mem_config),
        std::nullopt,
        output_mem_config);
    ComplexTensor grad_a =
        ttnn::operations::complex_binary::divide(grad_tensor, ttnn::conj(other, output_mem_config), output_mem_config);
    Tensor grad_a_r = where(condition_nan, std::nanf(""), ttnn::real(grad_a, output_mem_config), output_mem_config);
    Tensor grad_a_i = where(condition_nan, std::nanf(""), ttnn::imag(grad_a, output_mem_config), output_mem_config);
    grad_a = ComplexTensor({grad_a_r, grad_a_i});
    grad_a_r.deallocate();
    grad_a_i.deallocate();
    grad_tensor_res.emplace_back(grad_a);
    ComplexTensor neg_grad = ComplexTensor(
        {ttnn::neg(grad_tensor.real(), output_mem_config), ttnn::neg(grad_tensor.imag(), output_mem_config)});
    ComplexTensor grad_b = ttnn::operations::complex_binary::multiply(
        neg_grad,
        ttnn::conj(
            ttnn::operations::complex_binary::divide(
                ttnn::operations::complex_binary::divide(input_a, other, output_mem_config), other, output_mem_config),
            output_mem_config),
        output_mem_config);
    neg_grad.deallocate();
    Tensor grad_b_r = where(condition_nan, std::nanf(""), ttnn::real(grad_b, output_mem_config), output_mem_config);
    Tensor grad_b_i = where(condition_nan, std::nanf(""), ttnn::imag(grad_b, output_mem_config), output_mem_config);
    grad_b = ComplexTensor({grad_b_r, grad_b_i});
    grad_b_r.deallocate();
    grad_b_i.deallocate();
    condition_nan.deallocate();
    grad_tensor_res.emplace_back(grad_b);
    return grad_tensor_res;
}

std::vector<std::optional<Tensor>> mul_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& /*input_tensor_arg*/,
    float scalar,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result;
    if (!input_grad.has_value()) {
        input_grad = ttnn::empty_like(grad_tensor_arg);
    }
    ttnn::multiply(grad_tensor_arg, scalar, std::nullopt, output_mem_config, input_grad);
    result.push_back(input_grad);
    return result;
}

std::vector<std::optional<Tensor>> mul_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const Tensor& other_tensor_arg,
    const std::vector<bool>& are_required_outputs,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt, std::nullopt};
    operations::binary_backward::detail::preallocated_tensors_check(
        input_grad, other_grad, input_tensor_arg, other_tensor_arg, {are_required_outputs[0], are_required_outputs[1]});

    if (are_required_outputs.at(0)) {
        ttnn::multiply(grad_tensor_arg, other_tensor_arg, std::nullopt, output_mem_config, input_grad);
        result[0] = input_grad;
    }
    if (are_required_outputs.at(1)) {
        ttnn::multiply(grad_tensor_arg, input_tensor_arg, std::nullopt, output_mem_config, other_grad);
        result[1] = other_grad;
    }
    return result;
}

std::vector<ComplexTensor> mul_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& input_a,
    const ComplexTensor& other,
    const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor_res;
    ComplexTensor grad_a = ttnn::operations::complex_binary::multiply(
        grad_tensor, ttnn::conj(other, output_mem_config), output_mem_config);
    grad_tensor_res.emplace_back(grad_a);
    ComplexTensor grad_b = ttnn::operations::complex_binary::multiply(
        grad_tensor, ttnn::conj(input_a, output_mem_config), output_mem_config);
    grad_tensor_res.emplace_back(grad_b);
    return grad_tensor_res;
}

}  // namespace ttnn
