// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_unreduced_backward/moreh_nll_loss_unreduced_backward_op.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace {

inline void check_tensor(const Tensor &tensor, const std::string &op_name) {
    TT_FATAL(tensor.get_layout() == Layout::TILE, "{} only supports tiled layout.", op_name);
    TT_FATAL(tensor.get_dtype() == DataType::BFLOAT16, "{} only supports bfloat16.", op_name);
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE, "Operands to {} need to be on device!", op_name);
    TT_FATAL(
        tensor.buffer() != nullptr, "Operands to {} need to be allocated in buffers on device!", op_name);
}

inline void check_tensor(std::optional<Tensor> tensor, const std::string &op_name) {
    if (!tensor.has_value()) {
        return;
    }
    check_tensor(tensor.value(), op_name);
}

}  // namespace

namespace operations {
namespace primary {

void MorehNllLossUnreducedBackward::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Must have 2 input tensors");
    TT_FATAL(optional_input_tensors.size() == 1, "Must have 1 optional input tensors");

    const auto& target_tensor = input_tensors.at(0);
    const auto& output_grad_tensor = input_tensors.at(1);

    const auto& weight_tensor = optional_input_tensors.at(0);

    const auto& input_grad_tensor = output_tensors.at(0);

    TT_FATAL(target_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss_unreduced need to be on device!");
    TT_FATAL(target_tensor.buffer() != nullptr, "Operands to nll_loss_unreduced need to be allocated in buffers on device!");
    TT_FATAL((target_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss_unreduced must be tilized");
    TT_FATAL(target_tensor.get_dtype() == DataType::INT32);

    TT_FATAL(output_grad_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss_unreduced need to be on device!");
    TT_FATAL(
        output_grad_tensor.buffer() != nullptr, "Operands to nll_loss_unreduced need to be allocated in buffers on device!");
    TT_FATAL((output_grad_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss_unreduced must be tilized");
    TT_FATAL(output_grad_tensor.get_dtype() == DataType::BFLOAT16);

    if (input_grad_tensor.has_value()) {
        TT_FATAL(
            input_grad_tensor.value().storage_type() == StorageType::DEVICE,
            "Operands to nll_loss need to be on device!");
        TT_FATAL(
            input_grad_tensor.value().buffer() != nullptr,
            "Operands to nll_loss need to be allocated in buffers on device!");
        TT_FATAL(
            (input_grad_tensor.value().get_layout() == Layout::TILE), "target_tensor to nll_loss_unreduced must be tilized");
        TT_FATAL(input_grad_tensor.value().get_dtype() == DataType::BFLOAT16);
    }

    if (weight_tensor.has_value()) {
        TT_FATAL(
            weight_tensor.value().storage_type() == StorageType::DEVICE,
            "weight_tensor to nll_loss need to be on device!");
        TT_FATAL(
            weight_tensor.value().buffer() != nullptr,
            "weight_tensor to nll_loss need to be allocated in buffers on device!");
        TT_FATAL((weight_tensor.value().get_layout() == Layout::TILE), "weight_tensor to nll_loss_unreduced must be in tilized");
        TT_FATAL(weight_tensor.value().get_dtype() == DataType::BFLOAT16);
    }
}

std::vector<Shape> MorehNllLossUnreducedBackward::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // To calculate the output shape, we need the channel_size. However, the required tensors, target and output_grad,
    // do not contain the channel_size information.
    TT_FATAL(false, "moreh_nll_loss_unreduced_backward not support create output tensors.");
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> MorehNllLossUnreducedBackward::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(1).get_dtype(), Layout::TILE, this->memory_config);
}

operation::ProgramWithCallbacks MorehNllLossUnreducedBackward::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& target = input_tensors.at(0);
    const auto& output_grad = input_tensors.at(1);

    const auto& weight = optional_input_tensors.at(0);

    auto& input_grad = output_tensors.at(0);

    return {moreh_nll_loss_unreduced_backward_impl(
        target,
        weight,
        output_grad,
        input_grad,
        this->ignore_index,
        this->core_range,
        this->compute_kernel_config)};
}

Tensor moreh_nll_loss_unreduced_backward(
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const Tensor& output_grad_tensor,
    const std::optional<const Tensor> input_grad_tensor,
    const int32_t ignore_index,
    const MemoryConfig& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto device = output_grad_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {Tensor(
        operation::get_workers_for_op_output({target_tensor, output_grad_tensor}, {weight_tensor}))};

    operation::launch_op(
        [ignore_index, memory_config, all_cores, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehNllLossUnreducedBackward{
                    .ignore_index = ignore_index,
                    .memory_config = memory_config,
                    .core_range = all_cores,
                    .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {target_tensor, output_grad_tensor},
        output_tensors,
        {weight_tensor},
        {input_grad_tensor});

    return output_tensors.at(0);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
