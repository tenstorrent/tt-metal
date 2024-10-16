// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_step2_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_step2 {

MorehNllLossStep2DeviceOperation::program_factory_t MorehNllLossStep2DeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    return Factory{};
}

void MorehNllLossStep2DeviceOperation::validate_inputs(const operation_attributes_t& attributes,
                                                       const tensor_args_t& tensor_args) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& target_tensor = tensor_args.target_tensor;
    const std::optional<Tensor>& weight_tensor = tensor_args.weight_tensor;
    const std::optional<Tensor>& divisor_tensor = tensor_args.divisor_tensor;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "intput_tensor to nll_loss need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "intput_tensor to nll_loss need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.get_layout() == Layout::TILE), "intput_tensor to nll_loss must be tilized");
    TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16, "input tensor type must be bfloat16");

    TT_FATAL(target_tensor.storage_type() == StorageType::DEVICE, "target_tensor to nll_loss need to be on device!");
    TT_FATAL(target_tensor.buffer() != nullptr, "target_tensor to nll_loss need to be allocated in buffers on device!");
    TT_FATAL((target_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_FATAL(target_tensor.get_dtype() == DataType::INT32, "target tensor type must be int32");

    if (weight_tensor.has_value()) {
        TT_FATAL(weight_tensor.value().storage_type() == StorageType::DEVICE,
                 "weight_tensor to nll_loss need to be on device!");
        TT_FATAL(weight_tensor.value().buffer() != nullptr,
                 "weight_tensor to nll_loss need to be allocated in buffers on device!");
        TT_FATAL((weight_tensor.value().get_layout() == Layout::TILE),
                 "weight_tensor to nll_loss must be in row major layout");
        TT_FATAL(weight_tensor.value().get_dtype() == DataType::BFLOAT16, "weight tensor type must be bfloat16");
    }

    if (divisor_tensor.has_value()) {
        TT_FATAL(divisor_tensor.value().storage_type() == StorageType::DEVICE,
                 "divisor_tensor to nll_loss need to be on device!");
        TT_FATAL(divisor_tensor.value().buffer() != nullptr,
                 "divisor_tensor to nll_loss need to be allocated in buffers on device!");
        TT_FATAL((divisor_tensor.value().get_layout() == Layout::TILE), "divisor_tensor to nll_loss must be tilized");
        TT_FATAL(divisor_tensor.value().get_dtype() == DataType::BFLOAT16, "divisor tensor type must be bfloat16");
    }
}

void MorehNllLossStep2DeviceOperation::validate_on_program_cache_miss(const operation_attributes_t& attributes,
                                                                      const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

void MorehNllLossStep2DeviceOperation::validate_on_program_cache_hit(const operation_attributes_t& attributes,
                                                                     const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

MorehNllLossStep2DeviceOperation::shape_return_value_t MorehNllLossStep2DeviceOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.get_shape().value;
    auto input_shape_without_padding = input_shape.without_padding();
    auto input_rank = input_shape.rank();

    auto C = input_shape[1];

    auto dimensions_pads = std::vector<Padding::PadDimension>();
    std::vector<uint32_t> output_shape_vec;

    // Need extend 1d output to 2d, because TT not support 1d tensor
    if (input_rank == 2) {
        output_shape_vec.push_back(1);
        dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 0});
    }

    for (uint32_t dim = 0; dim < input_rank; dim++) {
        // skip C dim
        if (dim == 1) {
            continue;
        }

        output_shape_vec.push_back(input_shape_without_padding[dim]);
        dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 0});
    }

    // padding output
    {
        uint32_t output_rank = output_shape_vec.size();
        for (uint32_t dim = output_rank - 2; dim < output_rank; dim++) {
            uint32_t up32_shape = tt::round_up(output_shape_vec[dim], 32);
            uint32_t padding_back = up32_shape - output_shape_vec[dim];

            output_shape_vec[dim] = up32_shape;
            dimensions_pads[dim].back = padding_back;
        }
    }

    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    auto output_shape = Shape(tt::tt_metal::LegacyShape{output_shape_vec, padding});

    return output_shape;
}

MorehNllLossStep2DeviceOperation::tensor_return_value_t MorehNllLossStep2DeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    if (operation_attributes.reduction == NONE && tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }

    // In case reduction is 'sum' or 'mean' we need to create a tensor to save loss result and reduce it to
    // tensor_args.output_tensor using moreh_sum() operation
    auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
    auto dtype = tensor_args.input_tensor.get_dtype();
    Layout layout{Layout::TILE};
    auto device = tensor_args.input_tensor.device();

    return create_device_tensor(output_shape, dtype, layout, device, operation_attributes.memory_config);
}

std::tuple<MorehNllLossStep2DeviceOperation::operation_attributes_t, MorehNllLossStep2DeviceOperation::tensor_args_t>
MorehNllLossStep2DeviceOperation::invoke(const Tensor& input_tensor,
                                         const Tensor& target_tensor,
                                         const std::string reduction,
                                         const std::optional<Tensor>& weight_tensor,
                                         const std::optional<Tensor>& divisor_tensor,
                                         const std::optional<Tensor>& output_tensor,
                                         const int32_t ignore_index,
                                         const std::optional<ttnn::MemoryConfig>& memory_config,
                                         const DeviceComputeKernelConfig& compute_kernel_config) {
    return {operation_attributes_t{reduction,
                                   ignore_index < 0 ? std::numeric_limits<uint32_t>::max() : ignore_index,
                                   memory_config.value_or(input_tensor.memory_config()),
                                   compute_kernel_config},
            tensor_args_t{input_tensor, target_tensor, weight_tensor, divisor_tensor, output_tensor}};
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_step2
