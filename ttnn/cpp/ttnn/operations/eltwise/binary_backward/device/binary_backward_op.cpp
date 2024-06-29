// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_eager/tt_dnn/op_library/backward/backward_ops.cpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/binary_backward/device/binary_backward_op.hpp"
#include "tt_eager/tt_dnn/op_library/unpad/unpad_op.hpp"


namespace ttnn::operations::binary_backward {

namespace utils {


std::vector<ttnn::Tensor> _atan2_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    UnaryWithParam op1{UnaryOpType::SQUARE};
    UnaryWithParam op2{UnaryOpType::RECIP};
    Tensor recip_mul =
        ttnn::multiply(grad, tt::tt_metal::unary_chain(hypot(input, other), {op1, op2}, output_mem_config), std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(other, recip_mul, std::nullopt, output_mem_config);
    Tensor cond = ttnn::logical_and(eqz(input, output_mem_config), eqz(other, output_mem_config));
    grad_a = where(cond, t_nan, grad_a, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = ttnn::multiply(neg(input), recip_mul, std::nullopt, output_mem_config);
    grad_b = where(cond, t_nan, grad_b, output_mem_config);
    recip_mul.deallocate();
    cond.deallocate();
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}


std::vector<ttnn::Tensor> _embedding_bw(
    const Tensor& grad, const Tensor& input, const Tensor& weight, const MemoryConfig& output_mem_config) {
    TT_FATAL(input.get_dtype() == DataType::UINT32, "Input must be UINT32");
    TT_FATAL(
        grad.get_legacy_shape()[0] == 1 && grad.get_legacy_shape()[1] == 1,
        "First two dimensions for the grad must be 1");
    TT_FATAL(
        input.get_legacy_shape()[1] == 1 && input.get_legacy_shape()[2] == 1,
        "Only dim 0 && 3 for the input can be non 1");
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = embeddings(input, grad, false);
    grad_tensor.emplace_back(grad_a);

    return grad_tensor;
}

std::vector<std::optional<ttnn::Tensor>> _addalpha_bw(
    uint8_t queue_id,
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    float alpha,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result;

    if (are_required_outputs.at(0)) {
        if(input_grad.has_value()){
            assign(queue_id, grad, input_grad.value());
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

std::vector<ttnn::Tensor> _addalpha_bw_inter(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {

    auto result = _addalpha_bw(0, grad, input, other, alpha, output_mem_config, {true, true}, std::nullopt, std::nullopt);

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


std::vector<std::optional<Tensor>> _addalpha_bw_overload(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    float alpha,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
        uint8_t default_queue_id = 0;
    return _addalpha_bw(default_queue_id, grad, input, other, alpha, output_mem_config, are_required_outputs, input_grad, other_grad);
}

std::vector<ttnn::Tensor> _subalpha_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_b = ttnn::multiply(neg(grad, output_mem_config), alpha, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<ttnn::Tensor> _sub_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
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

std::vector<ttnn::Tensor> _add_bw_inter(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    auto result =  _add_bw(0, grad, input, other, output_mem_config, {true, true}, std::nullopt, std::nullopt);
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

std::vector<std::optional<Tensor>> _add_bw_overload(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    uint8_t default_queue_id = 0;
    return _addalpha_bw(default_queue_id, grad, input, other, 1.0f, output_mem_config, are_required_outputs, input_grad, other_grad);
}

std::vector<ttnn::Tensor> _xlogy_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad1_result = log(other, output_mem_config);
    Tensor zero_tensor = ttnn::operations::creation::zeros_like(other, other.get_dtype(), other.get_layout(), std::nullopt, output_mem_config);
    grad1_result = where(
        ttnn::logical_and(
            eqz(input, output_mem_config),
            ttnn::le(other, zero_tensor, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        zero_tensor,
        where(ltz(other, output_mem_config), std::nanf(" "), grad1_result, output_mem_config),
        output_mem_config);
    grad1_result =
        where(eq_unary(input, std::nanf(" "), output_mem_config), std::nanf(" "), grad1_result, output_mem_config);
    grad1_result = ttnn::multiply(grad, grad1_result, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad1_result);
    Tensor div_result = ttnn::multiply(input, recip(other, output_mem_config), std::nullopt, output_mem_config);
    Tensor grad2_result = ttnn::multiply(grad, div_result, std::nullopt, output_mem_config);
    grad2_result = where(
        eqz(other, output_mem_config),
        ttnn::multiply(sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config),
        grad2_result,
        output_mem_config);
    grad2_result =
        where(eq_unary(other, std::nanf(" "), output_mem_config), std::nanf(" "), grad2_result, output_mem_config);
    grad_tensor.emplace_back(grad2_result);
    return grad_tensor;
}


std::vector<ttnn::Tensor> _hypot_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result_recip = recip(hypot(input, other, output_mem_config), output_mem_config);
    Tensor grad_a =
        ttnn::multiply(grad, ttnn::multiply(input, result_recip, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b =
        ttnn::multiply(grad, ttnn::multiply(other, result_recip, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}


// torch reference
// - name: ldexp(Tensor self, Tensor other) -> Tensor
//   self: grad * 2^other
//   other: grad * self * ln(2) * (2^other)
// # M_LN2 = ln(2)= 0.693147180559945309417
std::vector<ttnn::Tensor> _ldexp_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tpow_o = ttnn::multiply(grad, rpow(other, 2.0, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(tpow_o);
    Tensor result = ttnn::multiply(input, ttnn::multiply(tpow_o, M_LN2, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
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
    const Tensor& grad, const Tensor& input_a, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor opexp =
        ttnn::add(ttnn::exp(ttnn::subtract(other, input_a, std::nullopt, output_mem_config), false, output_mem_config), 1, std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, recip(opexp, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    opexp = ttnn::add(ttnn::exp(ttnn::subtract(input_a, other, std::nullopt, output_mem_config), false, output_mem_config), 1, std::nullopt, output_mem_config);
    Tensor grad_b = ttnn::multiply(grad, recip(opexp, output_mem_config), std::nullopt, output_mem_config);
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
    const Tensor& grad, const Tensor& input_a, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor oppow =
        ttnn::add(rpow(ttnn::subtract(other, input_a, std::nullopt, output_mem_config), 2, output_mem_config), 1, std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, recip(oppow, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    oppow = ttnn::add(rpow(ttnn::subtract(input_a, other, std::nullopt, output_mem_config), 2, output_mem_config), 1, std::nullopt, output_mem_config);
    Tensor grad_b = ttnn::multiply(grad, recip(oppow, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}


std::vector<ttnn::Tensor> _squared_difference_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor difference = ttnn::subtract(input, other);
    Tensor grad_a = ttnn::multiply(ttnn::multiply(grad, difference, std::nullopt, output_mem_config), 2, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = ttnn::multiply(grad_a, -1, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> _binary_eq_bw(
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
        if(input_grad.has_value()){
            tt::tt_metal::zeros_like(cq_id, input, output_mem_config, input_grad);
        } else {
            input_grad = tt::tt_metal::zeros_like(cq_id, input, output_mem_config);
        }
        result.emplace_back(input_grad);
    } else {
        result.emplace_back(std::nullopt);
    }
    if (are_required_outputs.at(1)) {
        if(other_grad.has_value()){
            tt::tt_metal::zeros_like(cq_id, input, output_mem_config, other_grad);
        } else {
            other_grad = tt::tt_metal::zeros_like(cq_id, input, output_mem_config);
        }
        result.emplace_back(other_grad);
    } else {
        result.emplace_back(std::nullopt);
    }
    return std::move(result);
}

std::vector<ttnn::Tensor> _binary_eq_bw_inter(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    auto result = _binary_eq_bw(0, grad, input, other, output_mem_config, {true, true}, std::nullopt, std::nullopt);
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

std::vector<std::optional<Tensor>> _binary_eq_bw_overload(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    uint8_t default_queue_id = 0;
    return _binary_eq_bw(default_queue_id, grad, input, other, output_mem_config, are_required_outputs, input_grad, other_grad);
}

std::vector<Tensor> _binary_assign_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _concat_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, int dim, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    const tt::tt_metal::Shape start_index = {0, 0, 0, 0};
    const tt::tt_metal::Shape end_index = {
        input.get_legacy_shape()[0] - 1,
        input.get_legacy_shape()[1] - 1,
        input.get_legacy_shape()[2] - 1,
        input.get_legacy_shape()[3] - 1};

    Tensor grad_a = unpad(grad, start_index, end_index);
    grad_tensor.emplace_back(grad_a);

    tt::tt_metal::Shape start_index_2 = {0, 0, 0, 0};
    if (dim == 0) {
        start_index_2 = {input.get_legacy_shape()[0], 0, 0, 0};
    } else if (dim == 1) {
        start_index_2 = {input.get_legacy_shape()[0] - 1, input.get_legacy_shape()[1], 0, 0};
    } else if (dim == 2) {
        start_index_2 = {
            input.get_legacy_shape()[0] - 1, input.get_legacy_shape()[1] - 1, input.get_legacy_shape()[2], 0};
    } else if (dim == 3) {
        start_index_2 = {0, 0, 0, input.get_legacy_shape()[3]};
    }
    const tt::tt_metal::Shape end_index_2 = {
        grad.get_legacy_shape()[0] - 1,
        grad.get_legacy_shape()[1] - 1,
        grad.get_legacy_shape()[2] - 1,
        grad.get_legacy_shape()[3] - 1};
    Tensor grad_b = unpad(grad, start_index_2, end_index_2);
    grad_tensor.emplace_back(grad_b);

    return grad_tensor;
}

std::vector<Tensor> _binary_le_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = tt::tt_metal::zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    Tensor zero_input = tt::tt_metal::zeros_like(input, output_mem_config);
    grad_tensor.emplace_back(zero_input);
    return grad_tensor;
}

std::vector<Tensor> _rsub_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor = _subalpha_bw(grad, input, other, 1.0f, output_mem_config);
    std::swap(grad_tensor[0], grad_tensor[1]);
    return grad_tensor;
}

std::vector<Tensor> _bias_gelu_bw(
    const Tensor& grad,
    const Tensor& input_a,
    const Tensor& input_b,
    string approximate,
    const MemoryConfig& output_mem_config) {
    TT_FATAL((approximate == "none" || approximate == "tanh") && "Incorrect approximation type (expected 'none', 'tanh')");
    std::vector<Tensor> grad_tensor;
    Tensor input = ttnn::add(input_a, input_b);
    grad_tensor = tt::tt_metal::gelu_bw(grad, input, approximate = approximate, output_mem_config);
    grad_tensor.emplace_back(grad_tensor[0]);
    return grad_tensor;
}


std::vector<Tensor> _binary_gt_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    return _binary_le_bw(grad, input, other, output_mem_config);
}

std::vector<Tensor> _binary_ne_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    return _binary_le_bw(grad, input, other, output_mem_config);
}

std::vector<Tensor> _binary_ge_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    return _binary_le_bw(grad, input, other, output_mem_config);
}

// template parameter min_or_max = TRUE for MAX, FALSE for MIN
template <bool min_or_max>
std::vector<Tensor> _min_or_max_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    Tensor zeros_t = ttnn::operations::creation::zeros_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    std::vector<Tensor> grad_tensor;
    Tensor t_scale_grad = mul_unary(grad, 0.5, output_mem_config);
    Tensor t_sub = ttnn::subtract(other, input, std::nullopt, output_mem_config);
    Tensor t_sub_gtz = gtz(t_sub, output_mem_config);
    Tensor t_sub_eqz = eqz(t_sub, output_mem_config);
    Tensor t_sub_ltz = ltz(t_sub, output_mem_config);
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

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&)> get_function_type1(BinaryBackwardOpType OpType){
    switch (OpType) {
        case BinaryBackwardOpType::ATAN2_BW:
            return _atan2_bw;
        case BinaryBackwardOpType::EMBEDDING_BW:
            return _embedding_bw;
        case BinaryBackwardOpType::SUB_BW:
            return _sub_bw;
        case BinaryBackwardOpType::XLOGY_BW:
            return _xlogy_bw;
        case BinaryBackwardOpType::HYPOT_BW:
            return _hypot_bw;
        case BinaryBackwardOpType::LDEXP_BW:
            return _ldexp_bw;
        case BinaryBackwardOpType::LOGADDEXP_BW:
            return _logaddexp_bw;
        case BinaryBackwardOpType::LOGADDEXP2_BW:
            return _logaddexp2_bw;
        case BinaryBackwardOpType::SQUARED_DIFFERENCE_BW:
            return _squared_difference_bw;
        case BinaryBackwardOpType::ADD_BW:
            return _add_bw_inter;
        case BinaryBackwardOpType::BINARY_EQ_BW:
            return _binary_eq_bw_inter;
        case BinaryBackwardOpType::BINARY_ASSIGN_BW:
            return _binary_assign_bw;
        case BinaryBackwardOpType::BINARY_LE_BW:
            return _binary_le_bw;
        case BinaryBackwardOpType::RSUB_BW:
            return _rsub_bw;
        case BinaryBackwardOpType::BINARY_GT_BW:
            return _binary_gt_bw;
        case BinaryBackwardOpType::BINARY_NE_BW:
            return _binary_ne_bw;
        case BinaryBackwardOpType::BINARY_GE_BW:
            return _binary_ge_bw;
        case BinaryBackwardOpType::MIN_BW:
            return _min_or_max_bw<false>;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const Tensor&, float, const MemoryConfig&)> get_function_type1_w_float(BinaryBackwardOpType OpType){
    switch (OpType) {
        case BinaryBackwardOpType::SUBALPHA_BW:
            return _subalpha_bw;
        case BinaryBackwardOpType::ADDALPHA_BW:
            return _addalpha_bw_inter;
        case BinaryBackwardOpType::CONCAT_BW:
            return _concat_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const Tensor&, string, const MemoryConfig&)> get_function_type1_w_string(BinaryBackwardOpType OpType){
    switch (OpType) {
        case BinaryBackwardOpType::BIAS_GELU_BW:
            return _bias_gelu_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<std::optional<ttnn::Tensor>>(uint8_t , const Tensor&, const Tensor&, const Tensor&, float, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> get_function_type2(BinaryBackwardOpType OpType){
    switch (OpType) {
        case BinaryBackwardOpType::ADDALPHA_BW:
            return _addalpha_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<std::optional<ttnn::Tensor>>(const Tensor&, const Tensor&, const Tensor&, float, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> get_function_type2_wo_qid(BinaryBackwardOpType OpType){
    switch (OpType) {
        case BinaryBackwardOpType::ADDALPHA_BW:
            return _addalpha_bw_overload;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<std::optional<ttnn::Tensor>>(uint8_t , const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> get_function_type3(BinaryBackwardOpType OpType){
    switch (OpType) {
        case BinaryBackwardOpType::ADD_BW:
            return _add_bw;
        case BinaryBackwardOpType::BINARY_EQ_BW:
            return _binary_eq_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<std::optional<ttnn::Tensor>>(const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> get_function_type3_wo_qid(BinaryBackwardOpType OpType){
    switch (OpType) {
        case BinaryBackwardOpType::ADD_BW:
            return _add_bw_overload;
        case BinaryBackwardOpType::BINARY_EQ_BW:
            return _binary_eq_bw_overload;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}
}


}  // namespace ttnn::operations::binary
