// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <cstdint>

#include <tt-metalium/base_types.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_sum {
MorehSumOperation::program_factory_t MorehSumOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Case for int32
    const auto input_rank = tensor_args.input.padded_shape().rank();

    if (tensor_args.input.dtype() == DataType::INT32) {
        if (operation_attributes.dim == input_rank - 1) {
            return MorehSumWIntFactory{};
        }
        if (operation_attributes.dim == input_rank - 2) {
            return MorehSumHIntFactory{};
        }
        return MorehSumNCIntFactory{};
    }

    if (operation_attributes.dim == input_rank - 1) {
        return MorehSumWFactory{};
    }
    if (operation_attributes.dim == input_rank - 2) {
        return MorehSumHFactory{};
    }
    return MorehSumNCFactory{};
}

void validate_tensors(
    const MorehSumOperation::operation_attributes_t& operation_attributes,
    const MorehSumOperation::tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    check_tensor(input, "moreh_sum", "input", {DataType::BFLOAT16, DataType::INT32});
    check_tensor(output, "moreh_sum", "output", {DataType::BFLOAT16, DataType::INT32});

    validate_input_with_dim(input, operation_attributes.dim);

    if (output.has_value()) {
        validate_output_with_keepdim(input, output.value(), operation_attributes.dim, operation_attributes.keepdim);
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

MorehSumOperation::spec_return_value_t MorehSumOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return {tensor_args.output->tensor_spec()};
    }

    const auto& input = tensor_args.input;
    auto input_shape = input.logical_shape();
    if (input_shape.rank() < 2) {
        input_shape = input_shape.to_rank(2);
    }
    const auto input_rank = input_shape.rank();
    const bool is_tile_dim = (operation_attributes.dim == input_rank - 1 || operation_attributes.dim == input_rank - 2);
    log_debug(
        tt::LogOp,
        "{}:{} dim {}, keepdim {}",
        __func__,
        __LINE__,
        operation_attributes.dim,
        operation_attributes.keepdim);

    ttnn::Shape output_shape = input_shape;
    if (operation_attributes.keepdim) {
        // e.g. (2, 64, 64) with dim 0 to be (1, 64, 64)
        output_shape[operation_attributes.dim] = 1;
    } else {
        ttnn::SmallVector<uint32_t> shape;

        // e.g. (2, 64, 64) with dim 1 to be (2, 1[32], 64)
        // e.g. (2, 64, 64) with dim 0 to be (64, 64)
        for (int i = 0; i < input_rank; ++i) {
            bool is_reduced_dim = (i == operation_attributes.dim);
            if (is_reduced_dim && !is_tile_dim) {
                continue;
            }

            shape.push_back((is_reduced_dim && is_tile_dim) ? 1 : (input_shape[i]));
        }

        output_shape = ttnn::Shape(std::move(shape));
    }

    log_debug(tt::LogOp, "{}:{} output_shape {}", __func__, __LINE__, output_shape);
    return TensorSpec(
        output_shape,
        TensorLayout(
            tensor_args.input.dtype(), PageConfig(tensor_args.input.layout()), operation_attributes.memory_config));
};

MorehSumOperation::tensor_return_value_t MorehSumOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        log_debug(tt::LogOp, "{}:{} use output tensor", __func__, __LINE__);
        return {tensor_args.output.value()};
    }

    log_debug(tt::LogOp, "{}:{} create output tensor", __func__, __LINE__);
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::moreh::moreh_sum

namespace ttnn::prim {
ttnn::operations::moreh::moreh_sum::MorehSumOperation::tensor_return_value_t moreh_sum(
    const Tensor& input,
    int64_t dim,
    bool keepdim,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_sum::MorehSumOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        dim,
        keepdim,
        memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{input, output};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
