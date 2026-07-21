// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::moreh::moreh_norm {

std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p) {
    auto floored_p = std::floor(p);
    auto decimal = p - floored_p;
    bool p_is_negative = floored_p < 0.0f;
    if (p_is_negative) {
        floored_p = -floored_p;
    }
    return std::make_tuple(static_cast<uint32_t>(floored_p), decimal, p_is_negative);
}

inline void validate_input_tensor_with_dim(const Tensor& input, int64_t dim) {
    const auto input_rank = input.logical_shape().rank();
    TT_FATAL((dim >= 0 && dim <= ttnn::MAX_NUM_DIMENSIONS), "dim must be between 0 and {}.", ttnn::MAX_NUM_DIMENSIONS);
    TT_FATAL((dim < input_rank), "dim must be smaller than input tensor rank {}.", input_rank);
}

void MorehNormOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;
    const auto dim = operation_attributes.dim;
    check_tensor(input, "moreh_norm", "input");
    check_tensor(output, "moreh_norm", "output");
    validate_input_tensor_with_dim(input, dim);
    if (output.has_value()) {
        validate_output_with_keepdim(input, output.value(), dim, operation_attributes.keepdim);
    }
}

MorehNormOperation::program_factory_t MorehNormOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto dim = operation_attributes.dim;
    const auto input_rank = tensor_args.input.logical_shape().rank();
    if (dim == input_rank - 1) {
        return ProgramFactoryWOther{};
    }
    if (dim == input_rank - 2) {
        return ProgramFactoryHOther{};
    }
    return ProgramFactoryNCOther{};
}

void MorehNormOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehNormOperation::spec_return_value_t MorehNormOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->tensor_spec();
    }

    const auto& input_shape = tensor_args.input.logical_shape();
    const auto input_rank = input_shape.rank();
    const auto dim = operation_attributes.dim;
    const bool is_tile_dim = (dim == input_rank - 1 || dim == input_rank - 2);

    if (operation_attributes.keepdim) {
        auto shape = input_shape;
        shape[dim] = 1;
        return TensorSpec(
            shape,
            TensorLayout(tensor_args.input.dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
    }

    ttsl::SmallVector<uint32_t> shape;
    for (int i = 0; i < input_rank; ++i) {
        bool is_reduced_dim = (i == dim);
        if (is_reduced_dim && !is_tile_dim) {
            continue;
        }
        shape.push_back((is_reduced_dim && is_tile_dim) ? 1 : input_shape[i]);
    }
    return TensorSpec(
        ttnn::Shape(shape),
        TensorLayout(tensor_args.input.dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
};

MorehNormOperation::tensor_return_value_t MorehNormOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output = tensor_args.output;
    if (output.has_value()) {
        return output.value();
    }
    const auto& input = tensor_args.input;
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), input.device());
}

}  // namespace ttnn::operations::moreh::moreh_norm

namespace ttnn::prim {

ttnn::operations::moreh::moreh_norm::MorehNormOperation::tensor_return_value_t moreh_norm(
    const Tensor& input,
    float p,
    int64_t dim,
    bool keepdim,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_norm::MorehNormOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        p,
        dim,
        keepdim,
        memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4),
    };
    auto tensor_args = OperationType::tensor_args_t{input, output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
