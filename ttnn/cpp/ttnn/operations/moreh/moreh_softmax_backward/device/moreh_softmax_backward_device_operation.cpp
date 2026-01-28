// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_backward_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::moreh::moreh_softmax_backward {

#define L1_512KB (512 * 1024)

bool is_moreh_softmax_backward_w_small_available(const Tensor& tensor) {
    auto w = tensor.logical_shape()[-1];
    int32_t Wt = (w + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());

    auto tile_size = tt::tile_size(data_format);

    int32_t cb_usage = 0;        // bytes
    cb_usage += Wt * tile_size;  // output
    cb_usage += Wt * tile_size;  // output_grad
    cb_usage += 1 * tile_size;   // scaler
    cb_usage += 1 * tile_size;   // mask
    cb_usage += 2 * tile_size;   // input_grad
    cb_usage += Wt * tile_size;  // output * output_grad
    cb_usage += 1 * tile_size;   // reduce
    cb_usage += 1 * tile_size;   // dy - sum

    return (tensor.device()->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1) + cb_usage <= L1_512KB);
}

bool is_moreh_softmax_backward_h_small_available(const Tensor& tensor) {
    auto h = tensor.logical_shape()[-2];
    int32_t Ht = (h + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());

    auto tile_size = tt::tile_size(data_format);

    int32_t cb_usage = 0;        // bytes
    cb_usage += Ht * tile_size;  // output
    cb_usage += Ht * tile_size;  // output_grad
    cb_usage += 1 * tile_size;   // scaler
    cb_usage += 1 * tile_size;   // mask
    cb_usage += 2 * tile_size;   // input_grad
    cb_usage += Ht * tile_size;  // output * output_grad
    cb_usage += 1 * tile_size;   // reduce
    cb_usage += 1 * tile_size;   // dy - sum

    return (tensor.device()->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1) + cb_usage <= L1_512KB);
}

MorehSoftmaxBackwardOperation::program_factory_t MorehSoftmaxBackwardOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto strategy = get_parallelization_strategy(operation_attributes, tensor_args);

    switch (strategy) {
        case MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W: return MorehSoftmaxBackwardWSmallFactory{};
        case MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H: return MorehSoftmaxBackwardHSmallFactory{};
        case MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W: return MorehSoftmaxBackwardWLargeFactory{};
        case MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C: return MorehSoftmaxBackwardCLargeFactory{};
        default: return MorehSoftmaxBackwardHLargeFactory{};
    }
}

void MorehSoftmaxBackwardOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output_tensor;
    const auto& output_grad_tensor = tensor_args.output_grad_tensor;
    TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(output_grad_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(output_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL(output_grad_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL((output_tensor.layout() == Layout::TILE), "Output to softmax must be tilized");
    TT_FATAL((output_grad_tensor.layout() == Layout::TILE), "Output_grad to softmax must be tilized");
    TT_FATAL(
        output_tensor.dtype() == DataType::BFLOAT16 || output_tensor.dtype() == DataType::BFLOAT8_B,
        "Output_tensor dtype should be bfloat16 or bfloat8_b");
    TT_FATAL(
        output_grad_tensor.dtype() == DataType::BFLOAT16 || output_grad_tensor.dtype() == DataType::BFLOAT8_B,
        "Output_tensor_grad dtype should be bfloat16 or bfloat8_b");

    const auto rank = output_tensor.logical_shape().rank();
    const auto dim = operation_attributes.dim;
    TT_FATAL(dim >= 0 && dim < rank, "dim {} should be less than output tensor rank {}", dim, rank);
}

void MorehSoftmaxBackwardOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

void MorehSoftmaxBackwardOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

MorehSoftmaxBackwardOperation::spec_return_value_t MorehSoftmaxBackwardOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad_tensor.has_value()) {
        return tensor_args.input_grad_tensor->tensor_spec();
    }
    return TensorSpec(
        tensor_args.output_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.output_tensor.dtype(),
            tt::tt_metal::PageConfig(tensor_args.output_tensor.layout()),
            operation_attributes.memory_config));
}

MorehSoftmaxBackwardOperation::tensor_return_value_t MorehSoftmaxBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_grad_tensor = tensor_args.input_grad_tensor;
    if (input_grad_tensor.has_value()) {
        return input_grad_tensor.value();
    }

    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.output_tensor.device());
}

MorehSoftmaxBackwardOpParallelizationStrategy MorehSoftmaxBackwardOperation::get_parallelization_strategy(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output = tensor_args.output_tensor;
    const auto strategy = operation_attributes.strategy;
    const auto dim = operation_attributes.dim;

    auto rank = output.logical_shape().rank();
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
        TT_FATAL(
            strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H ||
                strategy == MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_H,
            "Invalid parallelization strategy. {} is not for dim H",
            strategy);

        if (strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H) {
            TT_FATAL(
                is_moreh_softmax_backward_h_small_available(output),
                "not enough circular buffer memory for {}",
                strategy);
        }
    } else if (rank - 1 == dim) {
        TT_FATAL(
            strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W ||
                strategy == MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W,
            "Invalid parallelization strategy. {} is not for dim W",
            strategy);

        if (strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W) {
            TT_FATAL(
                is_moreh_softmax_backward_w_small_available(output),
                "not enough circular buffer memory for {}",
                strategy);
        }
    } else {
        TT_FATAL(
            strategy == MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C,
            "Invalid parallelization strategy. large c is for dim 0 - (rank - 3)");
    }

    return strategy;
}

}  // namespace ttnn::operations::moreh::moreh_softmax_backward

namespace ttnn::prim {
ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOperation::tensor_return_value_t
moreh_softmax_backward(
    const Tensor& output_tensor,
    const Tensor& output_grad_tensor,
    uint32_t dim,
    const std::optional<Tensor>& input_grad_tensor,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp op,
    const ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy strategy,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        dim,
        op,
        strategy,
        memory_config.value_or(output_tensor.memory_config()),
        init_device_compute_kernel_config(
            output_grad_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{output_tensor, output_grad_tensor, input_grad_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
