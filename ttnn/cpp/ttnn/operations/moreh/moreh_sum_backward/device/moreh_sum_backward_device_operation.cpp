// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum_backward_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_sum_backward {
void MorehSumBackwardOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& input_grad = tensor_args.input_grad;

    const auto keepdim = operation_attributes.keepdim;
    const auto dims = operation_attributes.dims;

    // validate tensor
    check_tensor(output_grad, "moreh_sum_backward", "output_grad");
    check_tensor(input_grad, "moreh_sum_backward", " input_grad");

    if (!input.has_value()) {
        return;
    }

    check_tensor(input, "moreh_sum_backward", "input");
    const auto& input_shape = input.value().padded_shape();
    auto input_shape_wo_padding = input.value().logical_shape();
    auto input_rank = input_shape.rank();
    auto output_grad_shape_wo_padding = output_grad.logical_shape();

    // validate output_grad shape
    if (keepdim) {
        for (int i = 0; i < input_rank; ++i) {
            TT_FATAL(
                input_shape_wo_padding[i] >= output_grad_shape_wo_padding[i],
                "Input shape without padding[{}] ({}) must be >= output grad shape without padding[{}] ({})",
                i,
                input_shape_wo_padding[i],
                i,
                output_grad_shape_wo_padding[i]);
        }
    } else {
        std::vector<uint32_t> expected_output_grad_shape;
        std::vector<uint32_t> reduced_dims(input_rank, 0);
        for (auto dim : dims) {
            TT_FATAL(dim < input_rank, "dim {} < input_rank {}", dim, input_rank);
            reduced_dims[dim] = 1;
        }

        TT_FATAL(input_rank >= 2, "at least input_rank {} >= 2", input_rank);
        for (int i = 0; i < input_rank; ++i) {
            log_debug(tt::LogOp, "reduced_dims[{}] = {}", i, reduced_dims[i]);
            bool is_tile_dim = (i == input_rank - 1 || i == input_rank - 2);
            // batch dims
            if (reduced_dims[i] && !is_tile_dim) {
                continue;
            }
            uint32_t s = input_shape_wo_padding[i];
            // tile dims are not reduced
            if (reduced_dims[i] && is_tile_dim) {
                s = 1;
            }
            expected_output_grad_shape.push_back(s);
        }

        uint32_t expected_rank = expected_output_grad_shape.size();
        uint32_t rank = output_grad_shape_wo_padding.rank();
        TT_FATAL(expected_rank == rank, "expected_rank {} == rank {}", expected_rank, rank);
        for (int i = 0; i < rank; ++i) {
            TT_FATAL(
                expected_output_grad_shape[i] >= output_grad_shape_wo_padding[i],
                "Expected output grad shape[{}] ({}) must be >= output grad shape without padding[{}] ({})",
                i,
                expected_output_grad_shape[i],
                i,
                output_grad_shape_wo_padding[i]);
            log_debug(
                tt::LogOp,
                "rank {} expected_output_grad_shape {}, output_grad_shape_wo_padding {}",
                i,
                expected_output_grad_shape[i],
                output_grad_shape_wo_padding[i]);
        }
    }

    // validate input_grad shape
    if (input_grad.has_value()) {
        const auto& input_grad_shape = input_grad.value().padded_shape();
        TT_FATAL(input_shape == input_grad_shape, "both shape between input and input_grad should be the same");
    }
}

MorehSumBackwardOperation::program_factory_t MorehSumBackwardOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ProgramFactory{};
}

void MorehSumBackwardOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehSumBackwardOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehSumBackwardOperation::spec_return_value_t MorehSumBackwardOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad.has_value()) {
        return tensor_args.input_grad->tensor_spec();
    }
    TT_FATAL(tensor_args.input.has_value(), "input tensor should not be std::nullopt.");
    return TensorSpec(
        tensor_args.input->logical_shape(),
        TensorLayout(tensor_args.input->dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
};

MorehSumBackwardOperation::tensor_return_value_t MorehSumBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_grad = tensor_args.input_grad;
    if (input_grad.has_value()) {
        return input_grad.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input->device());
}

}  // namespace ttnn::operations::moreh::moreh_sum_backward

namespace ttnn::prim {
ttnn::operations::moreh::moreh_sum_backward::MorehSumBackwardOperation::tensor_return_value_t moreh_sum_backward(
    const Tensor& output_grad,
    const std::optional<Tensor>& input,
    tt::stl::Span<const int64_t> dims,
    bool keepdim,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_sum_backward::MorehSumBackwardOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        ttnn::SmallVector<int64_t>(dims.begin(), dims.end()),
        keepdim,
        memory_config.value_or(output_grad.memory_config()),
        init_device_compute_kernel_config(output_grad.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{output_grad, input, input_grad};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
