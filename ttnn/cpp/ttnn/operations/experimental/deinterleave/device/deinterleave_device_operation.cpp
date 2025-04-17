// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deinterleave_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::experimental::deinterleave {

void DeinterleaveToBatchOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    TT_FATAL(
        input.get_dtype() == DataType::BFLOAT16,
        "Deinterleave: input must be BFLOAT16");  // BFP8 requires untilizing/tilizing in the deinterleaving loop
    TT_FATAL(
        input.get_layout() == Layout::ROW_MAJOR,
        "Deinterleave: input must be ROW_MAJOR");  // TILE requires untilizing/tilizing in the deinterleaving loop
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Deinterleave: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Deinterleave: input must be allocated in buffer on device");
    TT_FATAL(
        input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Deinterleave: input must be HEIGHT_SHARDED");
    TT_FATAL(input.memory_config().shard_spec.has_value(), "Deinterleave: input must have shard_spec");
    TT_FATAL(
        input.memory_config().shard_spec.value().orientation == ShardOrientation::ROW_MAJOR,
        "Deinterleave: input must have ROW_MAJOR orientation");

    auto per_core_height = input.memory_config().shard_spec.value().shape[0] / operation_attributes.input_width;
    TT_FATAL(
        per_core_height >= operation_attributes.stride_hw[0],
        "Deinterleave: per_core_height {} must be larger than {}",
        per_core_height,
        operation_attributes.stride_hw[0]);
    TT_FATAL(
        per_core_height % (operation_attributes.stride_hw[0]) == 0,
        "Deinterleave: per_core_height {} must be div by {}",
        per_core_height,
        operation_attributes.stride_hw[0]);
    TT_FATAL(
        per_core_height * operation_attributes.input_width == input.memory_config().shard_spec.value().shape[0],
        "Deinterleave: per_core_height {} * input_width {} must be equal to input shard_spec shape {}",
        per_core_height,
        operation_attributes.input_width,
        input.memory_config().shard_spec.value().shape[0]);
}

DeinterleaveToBatchOperation::program_factory_t DeinterleaveToBatchOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactoryToBatch{};
}

void DeinterleaveToBatchOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void DeinterleaveToBatchOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

DeinterleaveToBatchOperation::spec_return_value_t DeinterleaveToBatchOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    tt::log_warning(tt::LogOp, "DeinterleaveLocal::compute_output_specs");

    auto tensor_spec = TensorSpec(
        input.get_logical_shape(),
        tt::tt_metal::TensorLayout(
            input.get_dtype(), tt::tt_metal::PageConfig(input.get_layout()), input.memory_config()));
    return tensor_spec;
};

DeinterleaveToBatchOperation::tensor_return_value_t DeinterleaveToBatchOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(operation_attributes, tensor_args);

    tt::log_warning(tt::LogOp, "DeinterleaveLocal::create_output_tensors");
    return create_device_tensor(spec, tensor_args.input.device());
}

std::tuple<DeinterleaveToBatchOperation::operation_attributes_t, DeinterleaveToBatchOperation::tensor_args_t>
DeinterleaveToBatchOperation::invoke(
    const Tensor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> stride_hw,
    const uint32_t barrier_threshold,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            input_height,
            input_width,
            stride_hw,
            barrier_threshold,
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
        },
        tensor_args_t{input},
    };
}

void DeinterleaveLocalOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    TT_FATAL(
        input.get_dtype() == DataType::BFLOAT16,
        "Deinterleave: input must be BFLOAT16");  // BFP8 requires untilizing/tilizing in the deinterleaving loop
    TT_FATAL(
        input.get_layout() == Layout::ROW_MAJOR,
        "Deinterleave: input must be ROW_MAJOR");  // TILE requires untilizing/tilizing in the deinterleaving loop
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Deinterleave: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Deinterleave: input must be allocated in buffer on device");
    TT_FATAL(
        input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Deinterleave: input must be HEIGHT_SHARDED");
    TT_FATAL(input.memory_config().shard_spec.has_value(), "Deinterleave: input must have shard_spec");
    TT_FATAL(
        input.memory_config().shard_spec.value().orientation == ShardOrientation::ROW_MAJOR,
        "Deinterleave: input must have ROW_MAJOR orientation");

    auto per_core_height = input.memory_config().shard_spec.value().shape[0] / operation_attributes.input_width;
    TT_FATAL(
        per_core_height >= operation_attributes.stride_hw[0],
        "Deinterleave: per_core_height {} must be larger than {}",
        per_core_height,
        operation_attributes.stride_hw[0]);
    TT_FATAL(
        per_core_height % (operation_attributes.stride_hw[0]) == 0,
        "Deinterleave: per_core_height {} must be div by {}",
        per_core_height,
        operation_attributes.stride_hw[0]);
    TT_FATAL(
        per_core_height * operation_attributes.input_width == input.memory_config().shard_spec.value().shape[0],
        "Deinterleave: per_core_height {} * input_width {} must be equal to input shard_spec shape {}",
        per_core_height,
        operation_attributes.input_width,
        input.memory_config().shard_spec.value().shape[0]);
}

DeinterleaveLocalOperation::program_factory_t DeinterleaveLocalOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactoryLocal{};
}

void DeinterleaveLocalOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void DeinterleaveLocalOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

DeinterleaveLocalOperation::spec_return_value_t DeinterleaveLocalOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    tt::log_warning(
        tt::LogOp, "DeinterleaveLocal::compute_output_specs; operation_attributes {}", operation_attributes);

    auto input_shard_shape = input.memory_config().shard_spec.value().shape;

    tt::log_warning(
        tt::LogOp, "DeinterleaveLocal::compute_output_specs; input.input_shard_shape {}", input_shard_shape);

    std::array<uint32_t, 2> output_shard_shape = {
        input.memory_config().shard_spec.value().shape[0] / operation_attributes.stride_hw[0] /
            operation_attributes.stride_hw[1],
        input.memory_config().shard_spec.value().shape[1]};

    tt::log_warning(tt::LogOp, "DeinterleaveLocal::output_shard_shape {}", output_shard_shape);

    auto output_shard_spec = tt::tt_metal::ShardSpec(
        input.shard_spec()->grid, output_shard_shape, input.memory_config().shard_spec.value().orientation);

    auto output_memory_config =
        ttnn::MemoryConfig(input.memory_config().memory_layout, ttnn::BufferType::L1, output_shard_spec);

    tt::log_warning(tt::LogOp, "DeinterleaveLocal::output_memory_config {}", output_memory_config);

    TT_FATAL(
        input.get_logical_shape()[0] * input.get_logical_shape()[1] * input.get_logical_shape()[2] ==
            operation_attributes.input_height * operation_attributes.input_width,
        "Deinterleave: input shape {} must be equal to operation_attributes input shape {}",
        input.get_logical_shape()[0] * input.get_logical_shape()[1] * input.get_logical_shape()[2],
        operation_attributes.input_height * operation_attributes.input_width);
    TT_FATAL(
        operation_attributes.input_height % operation_attributes.stride_hw[0] == 0,
        "Deinterleave: input height {} must be divisible by stride_hw[0] {}",
        operation_attributes.input_height,
        operation_attributes.stride_hw[0]);
    TT_FATAL(
        operation_attributes.input_width % operation_attributes.stride_hw[1] == 0,
        "Deinterleave: input width {} must be divisible by stride_hw[1] {}",
        operation_attributes.input_width,
        operation_attributes.stride_hw[1]);

    auto output_shape = ttnn::Shape(
        {input.get_logical_shape()[0],
         input.get_logical_shape()[1],
         input.get_logical_shape()[2] / (operation_attributes.stride_hw[0] * operation_attributes.stride_hw[1]),
         input.get_logical_shape()[3]});

    auto tensor_spec = TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input.get_dtype(), tt::tt_metal::PageConfig(input.get_layout()), output_memory_config));

    return tensor_spec;
};

DeinterleaveLocalOperation::tensor_return_value_t DeinterleaveLocalOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(operation_attributes, tensor_args);

    tt::log_warning(tt::LogOp, "DeinterleaveLocal::create_output_tensors");
    OptionalTensors output;
    for (auto i = 0; i < operation_attributes.stride_hw[0] * operation_attributes.stride_hw[1]; i++) {
        output.push_back(create_device_tensor(spec, tensor_args.input.device()));
    }

    return output;
}

std::tuple<DeinterleaveLocalOperation::operation_attributes_t, DeinterleaveLocalOperation::tensor_args_t>
DeinterleaveLocalOperation::invoke(
    const Tensor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> stride_hw,
    const uint32_t barrier_threshold,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            input_height,
            input_width,
            stride_hw,
            barrier_threshold,
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
        },
        tensor_args_t{input},
    };
}

}  // namespace ttnn::operations::experimental::deinterleave
