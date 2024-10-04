// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_device_operation.hpp"

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_mean {
void MorehMeanOperation::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& input = tensor_args.input;
    auto& output = tensor_args.output;

    TT_FATAL(
        (operation_attributes.dim >= 0 && operation_attributes.dim <= 7),
        "Invalid dimension value: {}. Expected a value between 0 and 7.",
        operation_attributes.dim);
    TT_FATAL(operation_attributes.divisor.has_value() == false, "divisor not supported yet.");

    tt::operations::primary::check_tensor(input, "moreh_mean", "input", {DataType::BFLOAT16});
    tt::operations::primary::check_tensor(output, "moreh_mean", "output", {DataType::BFLOAT16});

    tt::operations::primary::validate_input_with_dim(input, operation_attributes.dim);

    if (output.has_value()) {
        tt::operations::primary::validate_output_with_keepdim(
            input, output.value(), operation_attributes.dim, operation_attributes.keepdim);
    }
}
MorehMeanOperation::program_factory_t MorehMeanOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& input = tensor_args.input;

    auto rank = input.get_shape().rank();

    if (operation_attributes.dim + 1 == rank) {
        return MorehMeanWFactory{};
    } else if (operation_attributes.dim + 2 == rank) {
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

MorehMeanOperation::shape_return_value_t MorehMeanOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_shape = tensor_args.input.get_shape();
    auto output_shape = input_shape;
    auto input_rank = input_shape.rank();

    auto dim = operation_attributes.dim;

    if (operation_attributes.keepdim) {
        auto padding = output_shape.value.padding();
        if (dim + 1 == input_rank) {
            output_shape.value[dim] = tt::constants::TILE_WIDTH;
            padding[dim] = Padding::PadDimension{0, 31};
        } else if (dim + 2 == input_rank) {
            output_shape.value[dim] = tt::constants::TILE_HEIGHT;
            padding[dim] = Padding::PadDimension{0, 31};
        } else {
            output_shape.value[dim] = 1;
        }

        return Shape(tt::tt_metal::LegacyShape(output_shape.value, padding));
    }

    std::vector<uint32_t> shape;
    std::vector<Padding::PadDimension> pad_dimensions;
    const bool is_tile_dim = (dim == input_rank - 1 || dim == input_rank - 2);
    const std::size_t output_rank = (is_tile_dim) ? (input_rank) : (input_rank - 1);
    auto input_padding = input_shape.value.padding();

    // e.g. (2, 64, 64) with dim 1 to be (2, 1[32], 64)
    // e.g. (2, 64, 64) with dim 0 to be (64, 64)
    for (int i = 0; i < input_rank; ++i) {
        bool is_reduced_dim = (i == dim);
        if (is_reduced_dim && !is_tile_dim)
            continue;

        shape.push_back((is_reduced_dim && is_tile_dim) ? (tt::constants::TILE_HEIGHT) : (input_shape.value[i]));
        pad_dimensions.push_back((is_reduced_dim && is_tile_dim) ? (Padding::PadDimension{0, 31}) : (input_padding[i]));
    }

    auto padding = Padding(pad_dimensions, input_padding.pad_value());
    return Shape(tt::tt_metal::LegacyShape(shape, padding));
}

MorehMeanOperation::tensor_return_value_t MorehMeanOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& output = tensor_args.output;
    if (output.has_value()) {
        return {output.value()};
    }

    return create_device_tensor(
        compute_output_shapes(operation_attributes, tensor_args),
        tensor_args.input.get_dtype(),
        tensor_args.input.get_layout(),
        tensor_args.input.device(),
        operation_attributes.memory_config);
}

std::tuple<MorehMeanOperation::operation_attributes_t, MorehMeanOperation::tensor_args_t> MorehMeanOperation::invoke(
    const Tensor& input,
    const int64_t dim,
    const bool keepdim,
    const std::optional<uint32_t>& divisor,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        {dim,
         keepdim,
         divisor,
         memory_config.value_or(input.memory_config()),
         init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
        {input, output}};
}
}  // namespace ttnn::operations::moreh::moreh_mean
