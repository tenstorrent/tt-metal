// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv2d_new/device/conv2d_new_device_operation.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation.hpp"

namespace ttnn::prim {

Conv2dNewDeviceOperation::program_factory_t Conv2dNewDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    if (tensor_args.a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        return conv2d_new_detail::Conv2dWidthShardedDescriptorFactory{};
    }
    return conv2d_new_detail::Conv2dShardedDescriptorFactory{};
}

void Conv2dNewDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    Conv2dDeviceOperation::validate_on_program_cache_miss(args, tensor_args);
}

Conv2dNewDeviceOperation::spec_return_value_t Conv2dNewDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return Conv2dDeviceOperation::compute_output_specs(args, tensor_args);
}

Conv2dNewDeviceOperation::tensor_return_value_t Conv2dNewDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return Conv2dDeviceOperation::create_output_tensors(args, tensor_args);
}

tt::stl::hash::hash_t Conv2dNewDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Mix in our own type hash so we don't collide with the original Conv2dDeviceOperation cache entries.
    return tt::stl::hash::hash_objects_with_default_seed(
        tt::stl::hash::type_hash<Conv2dNewDeviceOperation>,
        Conv2dDeviceOperation::compute_program_hash(args, tensor_args));
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> Conv2dNewDeviceOperation::create_op_performance_model(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    return Conv2dDeviceOperation::create_op_performance_model(args, tensor_args, output_tensor);
}

Tensor conv2d_new(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    const ttnn::operations::sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    const Conv2dParallelizationConfig& parallelization_config,
    const Conv2dBlockConfig& block_config,
    const tt::tt_metal::MemoryConfig& memory_config,
    tt::tt_metal::DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool full_inner_dim,
    bool enable_activation_reuse,
    bool config_tensors_in_dram,
    std::optional<bool> force_split_reader) {
    using OperationType = Conv2dNewDeviceOperation;

    TT_FATAL(b.layout() == Layout::TILE, "Weights should be in TILE layout.");

    auto operation_attributes = OperationType::operation_attributes_t{
        .sliding_window_config = sliding_window_config,
        .output_channels = output_channels,
        .groups = groups,
        .untilize_out = untilize_out,
        .has_bias = bias.has_value(),
        .activation = activation,
        .parallelization_config = parallelization_config,
        .block_config = block_config,
        .memory_config = memory_config,
        .dtype = dtype,
        .input_tensor_shape = input_tensor_shape,
        .compute_kernel_config = compute_kernel_config,
        .enable_act_double_buffer = enable_act_double_buffer,
        .enable_weights_double_buffer = enable_weights_double_buffer,
        .full_inner_dim = full_inner_dim,
        .enable_activation_reuse = enable_activation_reuse,
        .config_tensors_in_dram = config_tensors_in_dram,
        .force_split_reader = force_split_reader,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .a = a,
        .b = b,
        .bias = bias,
    };

    auto* device = a.device();
    operation_attributes.pre_op_l1_allocation_size_bytes =
        device->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
