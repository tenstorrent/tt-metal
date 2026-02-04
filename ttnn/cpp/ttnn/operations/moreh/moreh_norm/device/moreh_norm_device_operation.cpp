// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
    const auto input_rank = input.padded_shape().rank();
    TT_FATAL(
        (dim >= 0 && dim <= tt::tt_metal::MAX_NUM_DIMENSIONS),
        "dim must be between 0 and {}.",
        tt::tt_metal::MAX_NUM_DIMENSIONS);
    TT_FATAL((dim < input_rank), "dim must be smaller than input tensor rank {}.", input_rank);
}

inline void validate_output_tensor_with_keepdim(const Tensor& input, const Tensor& output, int64_t dim, bool keepdim) {
    const auto& input_shape = input.padded_shape();
    const auto& input_shape_wo_padding = input.logical_shape();
    const auto input_rank = input_shape.rank();

    const auto& output_shape = output.padded_shape();
    const auto& output_shape_wo_padding = output.logical_shape();
    const auto output_rank = output_shape.rank();

    const bool is_tile_dim = (dim == input_rank - 1 || dim == input_rank - 2);

    if (keepdim) {
        TT_FATAL(input_rank == output_rank, "Input and output ranks must be equal when keepdim is true.");

        auto adjusted_input_shape = input_shape;
        auto adjusted_input_shape_wo_padding = input_shape_wo_padding;
        adjusted_input_shape[dim] = (is_tile_dim) ? tt::constants::TILE_HEIGHT : 1;
        adjusted_input_shape_wo_padding[dim] = 1;

        ttnn::SmallVector<uint32_t> input_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        ttnn::SmallVector<uint32_t> output_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        ttnn::SmallVector<uint32_t> input_dim_wo_padding(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        ttnn::SmallVector<uint32_t> output_dim_wo_padding(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);

        expand_to_max_dim(input_dim, adjusted_input_shape);
        expand_to_max_dim(output_dim, output_shape);
        expand_to_max_dim(input_dim_wo_padding, adjusted_input_shape_wo_padding);
        expand_to_max_dim(output_dim_wo_padding, output_shape_wo_padding);

        for (int i = 0; i < input_rank; ++i) {
            TT_FATAL(input_dim[i] == output_dim[i], "Input and output dimensions do not match at index {}.", i);
            TT_FATAL(
                input_dim_wo_padding[i] == output_dim_wo_padding[i],
                "Input and output dimensions without padding do not match at index {}.",
                i);
        }
    } else {
        TT_FATAL(!is_tile_dim, "Dimension {} should not be a tile dimension when keepdim is false.", dim);

        ttnn::SmallVector<uint32_t> expected_output_shape;
        ttnn::SmallVector<uint32_t> expected_output_shape_wo_padding;
        for (int i = 0; i < output_rank; ++i) {
            if (i == dim && !is_tile_dim) {
                expected_output_shape.push_back(1);
                expected_output_shape_wo_padding.push_back(1);
            }
            expected_output_shape.push_back(output_shape[i]);
            expected_output_shape_wo_padding.push_back(output_shape_wo_padding[i]);
        }

        for (int i = 0; i < input_rank; ++i) {
            if (i == dim) {
                continue;
            }
            TT_FATAL(
                input_shape[i] == expected_output_shape[i],
                "Input and expected output shapes do not match at index {}.",
                i);
            TT_FATAL(
                input_shape_wo_padding[i] == expected_output_shape_wo_padding[i],
                "Input and expected output shapes without padding do not match at index {}.",
                i);
        }
    }
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
        validate_output_tensor_with_keepdim(input, output.value(), dim, operation_attributes.keepdim);
    }
}

MorehNormOperation::program_factory_t MorehNormOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto dim = operation_attributes.dim;
    const auto input_rank = tensor_args.input.padded_shape().rank();
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

void MorehNormOperation::validate_on_program_cache_hit(
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

    ttnn::SmallVector<uint32_t> shape;
    for (int i = 0; i < input_rank; ++i) {
        bool is_reduced_dim = (i == dim);
        if (is_reduced_dim && !is_tile_dim) {
            continue;
        }
        shape.push_back(input_shape[i]);
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
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
    };
    auto tensor_args = OperationType::tensor_args_t{input, output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
