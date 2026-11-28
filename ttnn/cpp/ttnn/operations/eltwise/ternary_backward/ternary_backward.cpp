// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/ternary_backward/ternary_backward.hpp"

namespace ttnn {

namespace {

// Sum a gradient over the axes along which its operand was broadcast, so it comes back in the operand's own
// shape, as torch.autograd returns it (binary backward's reduce_grad_to_operand_shape does the same).
Tensor reduce_to_operand(const Tensor& grad, const Tensor& operand, const std::optional<MemoryConfig>& memory_config) {
    const auto& grad_shape = grad.logical_shape();
    const auto& operand_shape = operand.logical_shape();
    if (grad_shape == operand_shape) {
        return grad;
    }
    TT_FATAL(
        operand_shape.rank() <= grad_shape.rank(),
        "Operand shape {} is not broadcastable to gradient shape {}",
        operand_shape,
        grad_shape);
    const auto mc = memory_config.value_or(operand.memory_config());
    const size_t rank_diff = grad_shape.rank() - operand_shape.rank();
    ttsl::SmallVector<int> axes;
    for (size_t i = 0; i < grad_shape.rank(); ++i) {
        const uint32_t operand_dim = i < rank_diff ? 1u : operand_shape[i - rank_diff];
        TT_FATAL(
            operand_dim == 1u || operand_dim == grad_shape[i],
            "Operand shape {} is not broadcastable to gradient shape {}",
            operand_shape,
            grad_shape);
        if (operand_dim == 1u && grad_shape[i] != 1u) {
            axes.push_back(static_cast<int>(i));
        }
    }
    Tensor reduced = grad;
    if (!axes.empty()) {
        reduced = ttnn::sum(grad, std::variant<int, int64_t, ttsl::SmallVector<int>>{axes}, /*keepdim=*/true, mc);
    }
    if (reduced.logical_shape() != operand_shape) {
        reduced = ttnn::reshape(reduced, operand_shape, mc);
    }
    return reduced;
}

// Reduces grads[i] (when it was requested) to the shape of operands[i].
void reduce_to_operands(
    std::vector<std::optional<Tensor>>& grads,
    const std::vector<const Tensor*>& operands,
    const std::optional<MemoryConfig>& memory_config) {
    for (size_t i = 0; i < grads.size(); ++i) {
        if (grads[i].has_value()) {
            grads[i] = reduce_to_operand(grads[i].value(), *operands[i], memory_config);
        }
    }
}

}  // namespace

std::vector<std::optional<Tensor>> addcmul_bw(
    const Tensor& grad,
    const Tensor& input_a,
    const Tensor& tensor1,
    const Tensor& tensor2,
    float value,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<bool>& are_required_outputs) {
    TT_FATAL(are_required_outputs.size() == 3, "addcmul_bw expects are_required_outputs to hold 3 entries");
    auto output_mem_config = memory_config.value_or(input_a.memory_config());
    std::vector<std::optional<Tensor>> grad_tensor;
    grad_tensor.reserve(3);
    // Passthrough gradient, see #53874: no eltwise backward op relocates it.
    // output[0] keeps grad's own memory config rather than output_mem_config.
    grad_tensor.emplace_back(are_required_outputs[0] ? std::optional<Tensor>(grad) : std::nullopt);
    if (are_required_outputs[1]) {
        Tensor grad_a = ttnn::multiply(
            ttnn::multiply(grad, tensor2, std::nullopt, output_mem_config), value, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(std::move(grad_a));
    } else {
        grad_tensor.emplace_back(std::nullopt);
    }
    if (are_required_outputs[2]) {
        Tensor grad_b = ttnn::multiply(
            ttnn::multiply(grad, tensor1, std::nullopt, output_mem_config), value, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(std::move(grad_b));
    } else {
        grad_tensor.emplace_back(std::nullopt);
    }
    reduce_to_operands(grad_tensor, {&input_a, &tensor1, &tensor2}, memory_config);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> addcdiv_bw(
    const Tensor& grad,
    const Tensor& input_a,
    const Tensor& tensor1,
    const Tensor& tensor2,
    float value,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<bool>& are_required_outputs) {
    TT_FATAL(are_required_outputs.size() == 3, "addcdiv_bw expects are_required_outputs to hold 3 entries");
    auto output_mem_config = memory_config.value_or(input_a.memory_config());
    std::vector<std::optional<Tensor>> grad_tensor;
    grad_tensor.reserve(3);
    // Passthrough gradient, see #53874: no eltwise backward op relocates it.
    // output[0] keeps grad's own memory config rather than output_mem_config.
    grad_tensor.emplace_back(are_required_outputs[0] ? std::optional<Tensor>(grad) : std::nullopt);
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");
    // tensor2 is inverted once and reused for both gradients. The second gradient used to
    // form tensor2^2 and invert that, which loses the answer at both ends of the range --
    // infinity below |tensor2| = 1.0842e-19, zero above 2^63 (see the note in rdiv_bw).
    // Sharing the reciprocal is also one op cheaper than squaring and inverting.
    std::optional<Tensor> recip_tensor2;
    if (are_required_outputs[1] || are_required_outputs[2]) {
        recip_tensor2 = ttnn::reciprocal(tensor2, output_mem_config);
    }
    if (are_required_outputs[1]) {
        Tensor grad_a = ttnn::multiply(
            ttnn::multiply(grad, value, std::nullopt, output_mem_config),
            *recip_tensor2,
            std::nullopt,
            output_mem_config);
        grad_tensor.emplace_back(ttnn::where(
            ttnn::eqz(tensor2, output_mem_config),
            ttnn::where(ttnn::eqz(grad, output_mem_config), t_nan, t_inf, output_mem_config),
            grad_a,
            output_mem_config));
    } else {
        grad_tensor.emplace_back(std::nullopt);
    }
    if (are_required_outputs[2]) {
        Tensor tmp = ttnn::multiply(
            ttnn::multiply(ttnn::neg(grad, output_mem_config), value, std::nullopt, output_mem_config),
            tensor1,
            std::nullopt,
            output_mem_config);
        Tensor grad_b = ttnn::multiply(
            ttnn::multiply(tmp, *recip_tensor2, std::nullopt, output_mem_config),
            *recip_tensor2,
            std::nullopt,
            output_mem_config);
        grad_tensor.emplace_back(ttnn::where(
            ttnn::eqz(tensor2, output_mem_config),
            ttnn::where(ttnn::eqz(grad, output_mem_config), t_nan, -t_inf, output_mem_config),
            grad_b,
            output_mem_config));
    } else {
        grad_tensor.emplace_back(std::nullopt);
    }
    if (recip_tensor2.has_value()) {
        recip_tensor2->deallocate();
    }
    reduce_to_operands(grad_tensor, {&input_a, &tensor1, &tensor2}, memory_config);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> where_bw(
    const Tensor& grad,
    const Tensor& condition,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result;
    result.reserve(2);
    // A broadcast operand's gradient is computed at grad's shape, reduced to the operand's shape, then
    // copied into the preallocated output if there is one.
    auto finish = [&](const Tensor& operand, std::optional<Tensor>& preallocated, const auto& compute) {
        if (operand.logical_shape() == grad.logical_shape()) {
            if (preallocated.has_value()) {
                compute(preallocated);
            } else {
                preallocated = compute(std::nullopt);
            }
            return;
        }
        Tensor reduced = reduce_to_operand(compute(std::nullopt), operand, output_mem_config);
        if (preallocated.has_value()) {
            ttnn::assign(reduced, preallocated.value());
        } else {
            preallocated = std::move(reduced);
        }
    };
    if (are_required_outputs.at(0)) {
        finish(input, input_grad, [&](const std::optional<Tensor>& out) {
            return ttnn::where(condition, grad, 0.0f, output_mem_config, out);
        });
        result.emplace_back(std::move(input_grad));
    } else {
        result.emplace_back(std::nullopt);
    }
    if (are_required_outputs.at(1)) {
        finish(other, other_grad, [&](const std::optional<Tensor>& out) {
            return ttnn::where(condition, 0.0f, grad, output_mem_config, out);
        });
        result.emplace_back(std::move(other_grad));
    } else {
        result.emplace_back(std::nullopt);
    }
    return result;
}

// lerp(input, end, weight) = self: grad * (1 - weight), end: grad * weight
std::vector<std::optional<Tensor>> lerp_bw(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& end,
    const Tensor& weight,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::vector<bool>& are_required_outputs) {
    TT_FATAL(
        are_required_outputs.size() == 3,
        "lerp_bw with a tensor weight expects are_required_outputs to hold 3 entries");
    std::vector<std::optional<Tensor>> grad_tensor;
    grad_tensor.reserve(3);
    if (are_required_outputs[0]) {
        Tensor result_1 = ttnn::multiply(
            grad, ttnn::rsub(weight, 1.0f, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(std::move(result_1));
    } else {
        grad_tensor.emplace_back(std::nullopt);
    }
    if (are_required_outputs[1]) {
        Tensor result_2 = ttnn::multiply(grad, weight, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(std::move(result_2));
    } else {
        grad_tensor.emplace_back(std::nullopt);
    }
    if (are_required_outputs[2]) {
        Tensor zero = ttnn::multiply(
            grad, ttnn::subtract(end, input, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(std::move(zero));
    } else {
        grad_tensor.emplace_back(std::nullopt);
    }
    reduce_to_operands(grad_tensor, {&input, &end, &weight}, output_mem_config);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> lerp_bw(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& end,
    float weight,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::vector<bool>& are_required_outputs) {
    TT_FATAL(
        are_required_outputs.size() == 2,
        "lerp_bw with a scalar weight expects are_required_outputs to hold 2 entries");
    std::vector<std::optional<Tensor>> grad_tensor;
    grad_tensor.reserve(2);
    if (are_required_outputs[0]) {
        float sub_scalar = 1.0f - weight;
        grad_tensor.emplace_back(ttnn::multiply(grad, sub_scalar, std::nullopt, output_mem_config));
    } else {
        grad_tensor.emplace_back(std::nullopt);
    }
    if (are_required_outputs[1]) {
        grad_tensor.emplace_back(ttnn::multiply(grad, weight, std::nullopt, output_mem_config));
    } else {
        grad_tensor.emplace_back(std::nullopt);
    }
    reduce_to_operands(grad_tensor, {&input, &end}, output_mem_config);
    return grad_tensor;
}

}  // namespace ttnn
