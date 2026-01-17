// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_mean {
void MorehMeanOperation::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    TT_FATAL(
        (operation_attributes.dim >= 0 && operation_attributes.dim <= 7),
        "Invalid dimension value: {}. Expected a value between 0 and 7.",
        operation_attributes.dim);
    TT_FATAL(operation_attributes.divisor.has_value() == false, "divisor not supported yet.");

    check_tensor(input, "moreh_mean", "input", {DataType::BFLOAT16});
    check_tensor(output, "moreh_mean", "output", {DataType::BFLOAT16});

    validate_input_with_dim(input, operation_attributes.dim);

    if (output.has_value()) {
        validate_output_with_keepdim(input, output.value(), operation_attributes.dim, operation_attributes.keepdim);
    }
}
MorehMeanOperation::program_factory_t MorehMeanOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    auto rank = input.logical_shape().rank();

    if (operation_attributes.dim + 1 == rank) {
        return MorehMeanWFactory{};
    }
    if (operation_attributes.dim + 2 == rank) {
        return MorehMeanHFactory{};
    }
    return MorehMeanNCFactory{};
}

void MorehMeanOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

void MorehMeanOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

MorehMeanOperation::spec_return_value_t MorehMeanOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return {tensor_args.output->tensor_spec()};
    }

    auto input_shape = tensor_args.input.logical_shape();
    auto output_shape = input_shape;
    auto input_rank = input_shape.rank();

    auto dim = operation_attributes.dim;

    if (operation_attributes.keepdim) {
        output_shape[dim] = 1;
        return TensorSpec(
            output_shape,
            TensorLayout(
                tensor_args.input.dtype(), PageConfig(tensor_args.input.layout()), operation_attributes.memory_config));
    }

    ttnn::SmallVector<uint32_t> shape;
    const bool is_tile_dim = (dim == input_rank - 1 || dim == input_rank - 2);

    // e.g. (2, 64, 64) with dim 1 to be (2, 1[32], 64)
    // e.g. (2, 64, 64) with dim 0 to be (64, 64)
    for (int i = 0; i < input_rank; ++i) {
        bool is_reduced_dim = (i == dim);
        if (is_reduced_dim && !is_tile_dim) {
            continue;
        }

        shape.push_back((is_reduced_dim && is_tile_dim) ? 1 : input_shape[i]);
    }

    return TensorSpec(
        ttnn::Shape(std::move(shape)),
        TensorLayout(
            tensor_args.input.dtype(), PageConfig(tensor_args.input.layout()), operation_attributes.memory_config));
}

MorehMeanOperation::tensor_return_value_t MorehMeanOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output = tensor_args.output;
    if (output.has_value()) {
        return {output.value()};
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}
}  // namespace ttnn::operations::moreh::moreh_mean

namespace ttnn::prim {
ttnn::operations::moreh::moreh_mean::MorehMeanOperation::tensor_return_value_t moreh_mean(
    const Tensor& input,
    int64_t dim,
    bool keepdim,
    const std::optional<uint32_t>& divisor,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_mean::MorehMeanOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        dim,
        keepdim,
        divisor,
        memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{input, output};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
