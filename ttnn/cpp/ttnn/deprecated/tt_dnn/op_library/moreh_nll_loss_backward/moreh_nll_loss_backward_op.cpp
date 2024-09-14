// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward_op.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void MorehNllLossBackward::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_ASSERT(input_tensors.size() == 2, "Must have 2 input tensors");
    TT_ASSERT(optional_input_tensors.size() == 2, "Must have 2 optional input tensors");

    auto& target_tensor = input_tensors.at(0);
    auto& output_grad_tensor = input_tensors.at(1);
    auto& weight_tensor = optional_input_tensors.at(0);
    auto& divisor_tensor = optional_input_tensors.at(1);
    auto& input_grad_tensor = output_tensors.at(0);

    TT_ASSERT(target_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_ASSERT(target_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((target_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_ASSERT(target_tensor.get_dtype() == DataType::INT32);

    TT_ASSERT(output_grad_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_ASSERT(
        output_grad_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((output_grad_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_ASSERT(output_grad_tensor.get_dtype() == DataType::BFLOAT16);

    if (input_grad_tensor.has_value()) {
        TT_ASSERT(
            input_grad_tensor.value().storage_type() == StorageType::DEVICE,
            "Operands to nll_loss need to be on device!");
        TT_ASSERT(
            input_grad_tensor.value().buffer() != nullptr,
            "Operands to nll_loss need to be allocated in buffers on device!");
        TT_ASSERT(
            (input_grad_tensor.value().get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
        TT_ASSERT(input_grad_tensor.value().get_dtype() == DataType::BFLOAT16);
    }

    if (weight_tensor.has_value()) {
        TT_ASSERT(
            weight_tensor.value().storage_type() == StorageType::DEVICE,
            "weight_tensor to nll_loss need to be on device!");
        TT_ASSERT(
            weight_tensor.value().buffer() != nullptr,
            "weight_tensor to nll_loss need to be allocated in buffers on device!");
        TT_ASSERT((weight_tensor.value().get_layout() == Layout::TILE), "weight_tensor to nll_loss must be in tilized");
        TT_ASSERT(weight_tensor.value().get_dtype() == DataType::BFLOAT16);
    }

    if (divisor_tensor.has_value()) {
        TT_ASSERT(
            divisor_tensor.value().storage_type() == StorageType::DEVICE,
            "divisor_tensor to nll_loss need to be on device!");
        TT_ASSERT(
            divisor_tensor.value().buffer() != nullptr,
            "divisor_tensor to nll_loss need to be allocated in buffers on device!");
        TT_ASSERT((divisor_tensor.value().get_layout() == Layout::TILE), "divisor_tensor to nll_loss must be tilized");
        TT_ASSERT(divisor_tensor.value().get_dtype() == DataType::BFLOAT16);
    }
}

std::vector<Shape> MorehNllLossBackward::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // To calculate the output shape, we need the channel_size. However, the required tensors, target and output_grad,
    // do not contain the channel_size information.
    TT_ASSERT(false, "moreh_nll_loss_backward not support create output tensors.");
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> MorehNllLossBackward::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(1).get_dtype(), Layout::TILE, this->input_grad_mem_config);
}

operation::ProgramWithCallbacks MorehNllLossBackward::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& target = input_tensors.at(0);
    auto& output_grad = input_tensors.at(1);
    auto& weight = optional_input_tensors.at(0);
    auto& divisor = optional_input_tensors.at(1);
    auto& input_grad = output_tensors.at(0);

    return {moreh_nll_loss_backward_impl(
        target,
        weight,
        divisor,
        output_grad,
        input_grad,
        this->ignore_index,
        this->reduction_mean,
        this->core_range,
        this->compute_kernel_config)};
}

Tensor moreh_nll_loss_backward_(
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const Tensor& output_grad_tensor,
    const std::optional<const Tensor> input_grad_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig& input_grad_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto device = output_grad_tensor.device();
    auto grid_coord = DeviceComputeWithStorageGridSize(device);
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    auto kernel_config_val =
        init_device_compute_kernel_config(DeviceArch(device), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {Tensor(
        operation::get_workers_for_op_output({target_tensor, output_grad_tensor}, {weight_tensor, divisor_tensor}))};

    operation::launch_op(
        [ignore_index, reduction_mean, input_grad_mem_config, all_cores, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehNllLossBackward{
                    .ignore_index = ignore_index,
                    .reduction_mean = reduction_mean,
                    .input_grad_mem_config = input_grad_mem_config,
                    .core_range = all_cores,
                    .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {target_tensor, output_grad_tensor},
        output_tensors,
        {weight_tensor, divisor_tensor},
        {input_grad_tensor});

    return output_tensors.at(0);
}

Tensor moreh_nll_loss_backward(
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const Tensor& output_grad_tensor,
    const std::optional<const Tensor> input_grad_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig& input_grad_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return moreh_nll_loss_backward_(
        target_tensor,
        weight_tensor,
        divisor_tensor,
        output_grad_tensor,
        input_grad_tensor,
        ignore_index,
        reduction_mean,
        input_grad_mem_config,
        compute_kernel_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
