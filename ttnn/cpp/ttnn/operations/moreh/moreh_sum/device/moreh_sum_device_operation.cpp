// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum_device_operation.hpp"

#include <cstdint>

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_sum {
MorehSumOperation::program_factory_t MorehSumOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Case for int32
    const auto input_rank = tensor_args.input.get_shape().value.rank();

    if (tensor_args.input.dtype() == DataType::INT32) {
        if (operation_attributes.dim == input_rank - 1) {
            return MorehSumWIntFactory{};
        } else if (operation_attributes.dim == input_rank - 2) {
            return MorehSumHIntFactory{};
        } else {
            return MorehSumNCIntFactory{};
        }
    }

    if (operation_attributes.dim == input_rank - 1) {
        return MorehSumWFactory{};
    } else if (operation_attributes.dim == input_rank - 2) {
        return MorehSumHFactory{};
    } else {
        return MorehSumNCFactory{};
    }
}

void validate_tensors(
    const MorehSumOperation::operation_attributes_t& operation_attributes,
    const MorehSumOperation::tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    auto& output = tensor_args.output;

    tt::operations::primary::check_tensor(input, "moreh_sum", "input", {DataType::BFLOAT16, DataType::INT32});
    tt::operations::primary::check_tensor(output, "moreh_sum", "output", {DataType::BFLOAT16, DataType::INT32});

    tt::operations::primary::validate_input_with_dim(input, operation_attributes.dim);

    if (output.has_value()) {
        tt::operations::primary::validate_output_with_keepdim(
            input, output.value(), operation_attributes.dim, operation_attributes.keep_batch_dim);
    }
}

void MorehSumOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

void MorehSumOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

MorehSumOperation::shape_return_value_t MorehSumOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& input_shape = input.get_shape();
    const auto input_rank = input_shape.rank();
    const bool is_tile_dim = (operation_attributes.dim == input_rank - 1 || operation_attributes.dim == input_rank - 2);
    log_debug(
        tt::LogOp,
        "{}:{} dim {}, keep_batch_dim {}",
        __func__,
        __LINE__,
        operation_attributes.dim,
        operation_attributes.keep_batch_dim);

    Shape output_shape = input_shape;
    if (operation_attributes.keep_batch_dim) {
        auto shape = input_shape.value;
        auto padding = shape.padding();

        if (is_tile_dim) {
            // e.g. (2, 64, 64) with dim 1 to be (2, 1[32], 64)
            shape[operation_attributes.dim] = tt::constants::TILE_HEIGHT;
            padding[operation_attributes.dim] = Padding::PadDimension{0, 31};
        } else {
            // e.g. (2, 64, 64) with dim 0 to be (1, 64, 64)
            shape[operation_attributes.dim] = 1;
        }

        output_shape = Shape{tt::tt_metal::Shape(shape, padding)};
    } else {
        std::vector<uint32_t> shape;
        std::vector<Padding::PadDimension> pad_dimensions;
        const std::size_t output_rank = (is_tile_dim) ? (input_rank) : (input_rank - 1);
        auto input_padding = input_shape.value.padding();

        // e.g. (2, 64, 64) with dim 1 to be (2, 1[32], 64)
        // e.g. (2, 64, 64) with dim 0 to be (64, 64)
        for (int i = 0; i < input_rank; ++i) {
            bool is_reduced_dim = (i == operation_attributes.dim);
            if (is_reduced_dim && !is_tile_dim)
                continue;

            shape.push_back((is_reduced_dim && is_tile_dim) ? (tt::constants::TILE_HEIGHT) : (input_shape.value[i]));
            pad_dimensions.push_back(
                (is_reduced_dim && is_tile_dim) ? (Padding::PadDimension{0, 31}) : (input_padding[i]));
        }

        auto padding = Padding(pad_dimensions, input_padding.pad_value());
        output_shape = Shape{tt::tt_metal::Shape(shape, padding)};
    }

    log_debug(tt::LogOp, "{}:{} output_shape {}", __func__, __LINE__, output_shape);
    return {output_shape};
};

MorehSumOperation::tensor_return_value_t MorehSumOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        log_debug(tt::LogOp, "{}:{} use output tensor", __func__, __LINE__);
        return {tensor_args.output.value()};
    }

    log_debug(tt::LogOp, "{}:{} create output tensor", __func__, __LINE__);
    return create_device_tensor(
        compute_output_shapes(operation_attributes, tensor_args),
        tensor_args.input.get_dtype(),
        tensor_args.input.get_layout(),
        tensor_args.input.device(),
        operation_attributes.output_mem_config);
}

std::tuple<MorehSumOperation::operation_attributes_t, MorehSumOperation::tensor_args_t> MorehSumOperation::invoke(
    const Tensor& input,
    const int64_t dim,
    const bool keep_batch_dim,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        {dim, keep_batch_dim, output_mem_config.value_or(input.memory_config()), compute_kernel_config},
        {input, output}};
}
}  // namespace ttnn::operations::moreh::moreh_sum
