// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "batch_norm_utils.hpp"

namespace ttnn::operations::normalization {

namespace {
inline void check_tensor_BN(const Tensor& tensor, std::string_view name, std::uint32_t input_c_dim) {
    TT_FATAL(tensor.layout() == Layout::TILE, "batch_norm only supports tiled layout. Got: {}", tensor.layout());
    TT_FATAL(
        tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::FLOAT32,
        "batch_norm only supports bfloat16, float32. Got: {}",
        tensor.dtype());
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE,
        "Operands to batch_norm need to be on device! Got: {}",
        tensor.storage_type());
    TT_FATAL(tensor.buffer() != nullptr, "Operands to batch_norm need to be allocated in buffers on device!");
    TT_FATAL(tensor.logical_shape().rank() == 4, "batch_norm supports tensors of rank 4");
    TT_FATAL(tensor.logical_shape()[1] == input_c_dim, "{}[1] must be the same as input's channel size.", name);
}
}  // namespace

void BatchNormOperation::validate_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& [input, batch_mean, batch_var, weight, bias, output] = tensor_args;

    // input (N, C, H, W)
    auto C = input.logical_shape()[1];

    check_tensor_BN(input, "input_shape", C);
    check_tensor_BN(batch_mean, "batch_mean_shape", C);
    check_tensor_BN(batch_var, "batch_mean_shape", C);

    // output (N, C, H, W)
    if (output.has_value()) {
        check_tensor_BN(output.value(), "output_shape", C);
    }

    // weight (1, C, 1, 1)
    if (weight.has_value()) {
        check_tensor_BN(weight.value(), "weight_shape", C);
    }

    // bias (1, C, 1, 1)
    if (bias.has_value()) {
        check_tensor_BN(bias.value(), "bias_shape", C);
    }
}

BatchNormOperation::program_factory_t BatchNormOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return BatchNormFactory();
}

void BatchNormOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& [input, batch_mean, batch_var, weight, bias, output] = tensor_args;

    TT_FATAL(input.layout() == Layout::TILE, "Input tensor must be must be tilized");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Input tensor must be interleaved");
    TT_FATAL(
        operation_attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Output tensor to eltwise binary must be interleaved");

    TT_FATAL(batch_mean.layout() == Layout::TILE, "batch_mean tensor must be tilized");
    TT_FATAL(
        batch_mean.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "batch_mean tensor must be interleaved");

    TT_FATAL(batch_var.layout() == Layout::TILE, "batch_var tensor must be tilized");
    TT_FATAL(
        batch_var.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "batch_var tensor must be interleaved");

    if (weight.has_value()) {
        TT_FATAL(weight.value().layout() == Layout::TILE, "weight tensor must be tilized");
        TT_FATAL(
            weight.value().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "weight tensor must be interleaved");
    }

    if (bias.has_value()) {
        TT_FATAL(bias.value().layout() == Layout::TILE, "bias tensor must be tilized");
        TT_FATAL(
            bias.value().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "bias tensor must be interleaved");
    }

    BatchNormOperation::validate_on_program_cache_hit(operation_attributes, tensor_args);
};

void BatchNormOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

DataType BatchNormOperation::operation_attributes_t::get_dtype() const {
    return this->dtype.value_or(this->input_dtype);
}

BatchNormOperation::spec_return_value_t BatchNormOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto output_shape = tensor_args.input.logical_shape();
    return TensorSpec(
        output_shape,
        TensorLayout(operation_attributes.get_dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
}

BatchNormOperation::tensor_return_value_t BatchNormOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output;
    if (output_tensor.has_value()) {
        return output_tensor.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t BatchNormOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& [input, batch_mean, batch_var, weight, bias, output] = tensor_args;

    TT_FATAL(
        std::holds_alternative<DeviceStorage>(input.storage()),
        "Unexpected type {}",
        tt::stl::get_active_type_name_in_variant(input.storage()));

    // For input tensor
    auto base_tuple = std::make_tuple(attributes, input.dtype(), input.memory_config());

    // To extract (optional<DataType>, optional<MemoryConfig>) from optional tensors
    auto get_optional_tensor_info = [](const std::optional<const Tensor>& tensor_opt)
        -> std::tuple<std::optional<DataType>, std::optional<MemoryConfig>> {
        if (!tensor_opt.has_value()) {
            return std::make_tuple(std::nullopt, std::nullopt);
        }

        const auto& tensor = tensor_opt.value();
        return std::make_tuple(std::optional{tensor.dtype()}, std::optional{tensor.memory_config()});
    };

    auto args_tuple = std::tuple_cat(
        base_tuple,
        get_optional_tensor_info(batch_mean),
        get_optional_tensor_info(batch_var),
        get_optional_tensor_info(weight),
        get_optional_tensor_info(bias));

    // Apply the hash operation
    return std::apply(
        [](auto&&... args) {
            return operation::hash_operation<BatchNormOperation>(std::forward<decltype(args)>(args)...);
        },
        std::move(args_tuple));
}

tt::stl::hash::hash_t BatchNormOperation::operation_attributes_t::to_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(eps, memory_config, get_dtype(), compute_kernel_config);
}

}  // namespace ttnn::operations::normalization

namespace ttnn::prim {
ttnn::operations::normalization::BatchNormOperation::tensor_return_value_t batch_norm(
    const Tensor& input,
    const Tensor& batch_mean,
    const Tensor& batch_var,
    float eps,
    std::optional<Tensor> weight,
    std::optional<Tensor> bias,
    std::optional<Tensor> output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::normalization::BatchNormOperation;
    OperationType::operation_attributes_t operation_attributes{
        eps,
        memory_config.value_or(input.memory_config()),
        ttnn::operations::normalization::batch_norm::utils::resolve_compute_kernel_config(compute_kernel_config, input),
        input.dtype()};
    OperationType::tensor_args_t tensor_args{input, batch_mean, batch_var, std::move(weight), std::move(bias), std::move(output)};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
