// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_backward_device_operation.hpp"
#include <iostream>
#include "ttnn/tensor/tensor.hpp"


namespace ttnn::operations::moreh::moreh_softmax_backward {

#define L1_512KB (512 * 1024)

bool is_moreh_softmax_backward_w_small_available(const Tensor &tensor) {
    auto w = tensor.get_legacy_shape()[-1];
    int32_t Wt = (w + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.get_dtype());

    auto tile_size = tt::tt_metal::detail::TileSize(data_format);

    int32_t cb_usage = 0;        // bytes
    cb_usage += Wt * tile_size;  // output
    cb_usage += Wt * tile_size;  // output_grad
    cb_usage += 1 * tile_size;   // scaler
    cb_usage += 1 * tile_size;   // mask
    cb_usage += 2 * tile_size;   // input_grad
    cb_usage += Wt * tile_size;  // output * output_grad
    cb_usage += 1 * tile_size;   // reduce
    cb_usage += 1 * tile_size;   // dy - sum

    return (L1_UNRESERVED_BASE + cb_usage <= L1_512KB);
}

bool is_moreh_softmax_backward_h_small_available(const Tensor &tensor) {
    auto h = tensor.get_legacy_shape()[-2];
    int32_t Ht = (h + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.get_dtype());

    auto tile_size = tt::tt_metal::detail::TileSize(data_format);

    int32_t cb_usage = 0;        // bytes
    cb_usage += Ht * tile_size;  // output
    cb_usage += Ht * tile_size;  // output_grad
    cb_usage += 1 * tile_size;   // scaler
    cb_usage += 1 * tile_size;   // mask
    cb_usage += 2 * tile_size;   // input_grad
    cb_usage += Ht * tile_size;  // output * output_grad
    cb_usage += 1 * tile_size;   // reduce
    cb_usage += 1 * tile_size;   // dy - sum

    return (L1_UNRESERVED_BASE + cb_usage <= L1_512KB);
}

MorehSoftmaxBackwardOperation::program_factory_t MorehSoftmaxBackwardOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (operation_attributes.strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W) {
        return MorehSoftmaxBackwardWSmallFactory{};
    } else if(operation_attributes.strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H) {
        return MorehSoftmaxBackwardHSmallFactory{};
    } else if(operation_attributes.strategy == MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W) {
        return MorehSoftmaxBackwardWLargeFactory{};
    } else if(operation_attributes.strategy == MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C) {
        return MorehSoftmaxBackwardCLargeFactory{};
    } else {
        return MorehSoftmaxBackwardHLargeFactory{};
    }
}

void MorehSoftmaxBackwardOperation::validate_with_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& output_tensor = tensor_args.output_tensor;
    auto& output_grad_tensor = tensor_args.output_grad_tensor;
    TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(output_grad_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(output_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL(output_grad_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL((output_tensor.get_layout() == Layout::TILE), "Output to softmax must be tilized");
    TT_FATAL((output_grad_tensor.get_layout() == Layout::TILE), "Output_grad to softmax must be tilized");
    TT_FATAL(output_tensor.get_dtype() == DataType::BFLOAT16 || output_tensor.get_dtype() == DataType::BFLOAT8_B,
        "Output_tensor dtype should be bfloat16 or bfloat8_b");
    TT_FATAL(output_grad_tensor.get_dtype() == DataType::BFLOAT16 || output_grad_tensor.get_dtype() == DataType::BFLOAT8_B,
        "Output_tensor_grad dtype should be bfloat16 or bfloat8_b");

    // validate parameters
    auto rank = output_tensor.get_legacy_shape().rank();

    TT_FATAL(
        operation_attributes.dim >= 0 && operation_attributes.dim < rank,
        "dim {} should be less than output tensor rank {}", operation_attributes.dim, rank);
    if (!tensor_args.input_grad_tensor.has_value()) {
        // If the user decided to not use any optional output tensors, then this would be empty or would be a nullptr.
        return;
    }
}

void MorehSoftmaxBackwardOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_with_output_tensors(operation_attributes, tensor_args);
}

void MorehSoftmaxBackwardOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_with_output_tensors(operation_attributes, tensor_args);
}

MorehSoftmaxBackwardOperation::shape_return_value_t MorehSoftmaxBackwardOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.get_shape();
}


MorehSoftmaxBackwardOperation::tensor_return_value_t MorehSoftmaxBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad_tensor.has_value()) {
        return {tensor_args.input_grad_tensor.value()};
    }
    const auto& input_grad_tensor_shape = tensor_args.output_tensor.get_legacy_shape();
    return create_device_tensor(
        input_grad_tensor_shape,
        tensor_args.output_tensor.tensor_attributes->dtype,
        tensor_args.output_tensor.tensor_attributes->layout,
        tensor_args.output_tensor.device(),
        operation_attributes.output_memory_config);
}

std::tuple<MorehSoftmaxBackwardOperation::operation_attributes_t, MorehSoftmaxBackwardOperation::tensor_args_t>
MorehSoftmaxBackwardOperation::invoke(
    const Tensor &output_tensor,
    const Tensor &output_grad_tensor,
    const uint32_t dim,
    const std::optional<Tensor> &input_grad_tensor,
    const MorehSoftmaxBackwardOp op,
    const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
    const std::optional<MemoryConfig> output_memory_config,
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto parallelization_strategy = MorehSoftmaxBackwardOperation::get_parallelization_strategy(
        output_tensor, output_grad_tensor, dim, input_grad_tensor, op, strategy, output_memory_config, compute_kernel_config);
    return {
        operation_attributes_t{
            dim, op, parallelization_strategy, output_memory_config.value_or(output_tensor.memory_config()), compute_kernel_config
        },
        tensor_args_t{
            output_tensor, output_grad_tensor, input_grad_tensor
        }
    };
}

MorehSoftmaxBackwardOpParallelizationStrategy MorehSoftmaxBackwardOperation::get_parallelization_strategy(
    const Tensor &output_tensor,
    const Tensor &output_grad_tensor,
    const uint32_t dim,
    const std::optional<Tensor> &input_grad_tensor,
    const MorehSoftmaxBackwardOp op,
    const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
    const std::optional<MemoryConfig> output_memory_config,
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto& output = output_tensor;

    auto rank = output.get_legacy_shape().rank();

    if (strategy == MorehSoftmaxBackwardOpParallelizationStrategy::NONE) {
        if (rank - 1 == dim) {
            if (is_moreh_softmax_backward_w_small_available(output)) {
                return MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W;
            }
            return MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W;
        }
        if (rank - 2 == dim) {
            if (is_moreh_softmax_backward_h_small_available(output)) {
                return MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H;
            }
            return MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_H;
        }
        return MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C;
    }

    if (rank - 2 == dim) {
        TT_ASSERT(
            strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H ||
                strategy == MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_H,
            "Invalid parallelization strategy. {} is not for dim H", strategy);

        if (strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H) {
            TT_ASSERT(
                is_moreh_softmax_backward_h_small_available(output),
                "not enough circular buffer memory for {}", strategy);
        }
    } else if (rank - 1 == dim) {
        TT_ASSERT(
            strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W ||
                strategy == MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W,
            "Invalid parallelization strategy. {} is not for dim W", strategy);

        if (strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W) {
            TT_ASSERT(
                is_moreh_softmax_backward_w_small_available(output),
                "not enough circular buffer memory for {}", strategy);
        }
    } else {
        TT_ASSERT(
            strategy == MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C,
            "Invalid parallelization strategy. large c is for dim 0 - (rank - 3)");
    }

    return strategy;
}

}
