// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dropout_device_operation.hpp"

#include <magic_enum/magic_enum.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::dropout {

DropoutDeviceOperation::program_factory_t DropoutDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (args.use_per_device_seed) {
        return program::DropoutMeshWorkloadFactory{};
    } else {
        return program::DropoutProgramFactory{};
    }
}

void DropoutDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void DropoutDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;

    auto out_memory_config = args.output_memory_config;
    auto output_datatype = args.output_dtype;
    if (preallocated_output_tensor.has_value()) {
        out_memory_config = preallocated_output_tensor->memory_config();
        output_datatype = preallocated_output_tensor->get_dtype();
    }
    TT_FATAL(
        output_datatype == input_tensor.dtype(),
        "Dropout operation requires input and output data types to match. Input data type: {}, Output data type: {}",
        static_cast<int>(input_tensor.dtype()),
        static_cast<int>(output_datatype));

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Dropout operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to dropout need to be allocated in buffers on the device. Buffer is null.");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "Dropout operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        static_cast<int>(input_tensor.memory_config().memory_layout()),
        static_cast<int>(out_memory_config.memory_layout()));

    if (!input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.get_layout() == Layout::TILE,
            "Dropout operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
            "tensor layout: {}",
            static_cast<int>(input_tensor.get_layout()));

        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Dropout operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
            "memory layout: `{}`",
            static_cast<int>(input_tensor.memory_config().memory_layout()));
    }

    if (preallocated_output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto preallocated_output_shape = preallocated_output_tensor.value().get_logical_shape();
        TT_FATAL(
            preallocated_output_shape == computed_output_shape,
            "When preallocted output tensor is used, Dropout operation requires its shape to match the computed "
            "shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            preallocated_output_shape);

        if (!input_tensor.is_sharded()) {
            TT_FATAL(
                (preallocated_output_tensor.value().get_layout() == Layout::TILE),
                "Dropout operation requires output tensor to be in Tile layout when working with non-sharded tensor.");
        }
    }
}

spec_return_value_t DropoutDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->get_tensor_spec();
    }

    auto output_layout = Layout::TILE;
    if (args.output_memory_config.is_sharded()) {
        output_layout = tensor_args.input.get_layout();
    }

    const auto output_shape = tensor_args.input.logical_shape();
    return TensorSpec(output_shape, TensorLayout(args.output_dtype, output_layout, args.output_memory_config));
}

tensor_return_value_t DropoutDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t DropoutDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.get_padded_shape();
    auto args_without_seed = args;
    args_without_seed.seed = 0;
    auto program_factory = select_program_factory(args, tensor_args);
    operation::Hash hash = operation::hash_operation<DropoutDeviceOperation>(
        args_without_seed,
        program_factory.index(),
        input_tensor.dtype(),
        std::get<DeviceStorage>(input_tensor.storage()).memory_config(),
        input_shape.volume());

    return hash;
}

std::tuple<DropoutDeviceOperation::operation_attributes_t, DropoutDeviceOperation::tensor_args_t>
DropoutDeviceOperation::invoke(
    const Tensor& input,
    float prob,
    float scale,
    uint32_t seed,
    bool use_per_device_seed,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output) {
    return {
        operation_attributes_t{
            .output_dtype = output_dtype,
            .output_memory_config = output_memory_config,
            .seed = seed,
            .use_per_device_seed = use_per_device_seed,
            .prob = prob,
            .scale = scale,
        },
        tensor_args_t{.input = input, .preallocated_output = preallocated_output}};
}

}  // namespace ttnn::operations::experimental::dropout
