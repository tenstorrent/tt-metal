// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_dot_backward/moreh_dot_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace tt {

using namespace constants;

namespace operations {
namespace primary {

void grad_tensor_validate(const Tensor& tensor, const Tensor& grad_tensor) {
    const auto& tensor_shape = tensor.get_legacy_shape().without_padding();
    const auto& grad_tensor_shape = grad_tensor.get_legacy_shape().without_padding();
    TT_ASSERT(tensor_shape == grad_tensor_shape);
    TT_ASSERT(grad_tensor.storage_type() == StorageType::DEVICE, "Operands to dot backward need to be on device!");
    TT_ASSERT(grad_tensor.device() == tensor.device(), "Operands to dot backward need to be on the same device!");
    TT_ASSERT(grad_tensor.buffer() != nullptr, "Operands to dot backward need to be allocated in buffers on device!");
}

void MorehDotBackward::validate(
    const std::vector<Tensor>& inputs, const std::vector<std::optional<const Tensor>>& optional_inputs) const {
    // validate inputs
    const auto& output_grad = inputs.at(0);
    const auto& input = inputs.at(1);
    const auto& other = inputs.at(2);

    TT_ASSERT(is_scalar(output_grad));
    TT_ASSERT(is_1d_tensor(input));
    TT_ASSERT(is_1d_tensor(other));
    TT_ASSERT(is_same_shape(input, other));

    TT_ASSERT(
        input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::BFLOAT8_B, "Unsupported data format");
    TT_ASSERT(
        output_grad.storage_type() == StorageType::DEVICE and input.storage_type() == StorageType::DEVICE and
            other.storage_type() == StorageType::DEVICE,
        "Operands to dot backward need to be on device!");
    TT_ASSERT(
        output_grad.device() == input.device() and input.device() == other.device(),
        "Operands to dot backward need to be on the same device!");
    TT_ASSERT(
        output_grad.buffer() != nullptr and input.buffer() != nullptr and other.buffer() != nullptr,
        "Operands to dot backward need to be allocated in buffers on device!");

    // validate optional inputs
    const auto& input_grad = optional_inputs.at(0);
    const auto& other_grad = optional_inputs.at(1);
    if (input_grad) {
        grad_tensor_validate(input, input_grad.value());
    }

    if (other_grad) {
        grad_tensor_validate(other, other_grad.value());
    }
}

std::vector<Shape> MorehDotBackward::compute_output_shapes(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

std::vector<Tensor> MorehDotBackward::create_output_tensors(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

operation::ProgramWithCallbacks MorehDotBackward::create_program(
    const std::vector<Tensor>& inputs,
    const std::vector<std::optional<const Tensor>>& optional_inputs,
    std::vector<Tensor>& outputs) const {
    const auto& output_grad = inputs.at(0);
    const auto& input = inputs.at(1);
    const auto& other = inputs.at(2);
    const auto& input_grad = optional_inputs.at(0);
    const auto& other_grad = optional_inputs.at(1);
    return moreh_dot_backward_single_core(output_grad, input, other, input_grad, other_grad);
}

std::vector<std::optional<Tensor>> moreh_dot_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    std::optional<const Tensor> input_grad,
    std::optional<const Tensor> other_grad,
    const MemoryConfig& mem_config) {
    std::vector<Tensor> dummy_output_tensors = {
        Tensor(operation::get_workers_for_op_output({output_grad, input, other}, {input_grad, other_grad}))};

    operation::launch_op(
        [](const std::vector<Tensor>& input_tensors,
           const std::vector<std::optional<const Tensor>>& optional_input_tensors,
           const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(MorehDotBackward{}, input_tensors, optional_input_tensors, optional_output_tensors);
        },
        {output_grad, input, other},
        dummy_output_tensors,
        {input_grad, other_grad});

    std::vector<std::optional<Tensor>> outputs(2);
    if (input_grad) {
        outputs[0] = input_grad;
    }

    if (other_grad) {
        outputs[1] = other_grad;
    }

    return outputs;
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
