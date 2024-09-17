// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_softmax_backward/moreh_softmax_backward_op.hpp"

#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void MorehSoftmaxBackward::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    // validate input tensor
    auto& output_tensor = input_tensors.at(0);
    auto& output_grad_tensor = input_tensors.at(1);

    TT_ASSERT(output_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_ASSERT(output_grad_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_ASSERT(output_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_ASSERT(output_grad_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_ASSERT((output_tensor.get_layout() == Layout::TILE), "Output to softmax must be tilized");
    TT_ASSERT((output_grad_tensor.get_layout() == Layout::TILE), "Output_grad to softmax must be tilized");
    TT_ASSERT(output_tensor.get_dtype() == DataType::BFLOAT16 || output_tensor.get_dtype() == DataType::BFLOAT8_B);
    TT_ASSERT(
        output_grad_tensor.get_dtype() == DataType::BFLOAT16 || output_grad_tensor.get_dtype() == DataType::BFLOAT8_B);

    // validate parameters
    auto rank = output_tensor.get_legacy_shape().rank();

    TT_ASSERT(
        this->dim >= 0 && this->dim < rank,
        "dim {} should be less than output tensor rank {}", this->dim, rank);

    if (output_tensors.empty() || !output_tensors.at(0).has_value()) {
        // If the user decided to not use any optional output tensors, then this would be empty or would be a nullptr.
        return;
    }
    TT_ASSERT(input_tensors.size() == 2, "Must have 2 input tensors");
    TT_ASSERT(output_tensors.size() == 1, "Must have 1 output tensors");
}

std::vector<tt::tt_metal::LegacyShape> MorehSoftmaxBackward::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> MorehSoftmaxBackward::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }
    const auto& output_shape = input_tensors.at(0).get_legacy_shape();

    return {operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config)};
}

operation::ProgramWithCallbacks MorehSoftmaxBackward::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto& output = input_tensors.at(0);
    auto& output_grad = input_tensors.at(1);
    auto& input_grad = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy) {
        case MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W:
            return {moreh_softmax_backward_w_small(
                output, output_grad, input_grad, this->core_range, this->op, this->compute_kernel_config)};
        case MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H:
            return {moreh_softmax_backward_h_small(
                output, output_grad, input_grad, this->core_range, this->op, this->compute_kernel_config)};
        case MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W:
            return {moreh_softmax_backward_w_large(
                output, output_grad, input_grad, this->core_range, this->op, this->compute_kernel_config)};
        case MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_H:
            return {moreh_softmax_backward_h_large(
                output, output_grad, input_grad, this->core_range, this->op, this->compute_kernel_config)};
        case MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C:
            return {moreh_softmax_backward_c_large(
                output, output_grad, input_grad, this->dim, this->core_range, this->op, this->compute_kernel_config)};
        case MorehSoftmaxBackwardOpParallelizationStrategy::NONE:
        default: break;
    }

    return {moreh_softmax_backward_h_large(
        output, output_grad, input_grad, this->core_range, this->op, this->compute_kernel_config)};
}

MorehSoftmaxBackwardOpParallelizationStrategy MorehSoftmaxBackward::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    auto& output = input_tensors.at(0);

    auto rank = output.get_legacy_shape().rank();

    if (this->strategy == MorehSoftmaxBackwardOpParallelizationStrategy::NONE) {
        if (rank - 1 == this->dim) {
            if (is_moreh_softmax_backward_w_small_available(output)) {
                return MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W;
            }
            return MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W;
        }
        if (rank - 2 == this->dim) {
            if (is_moreh_softmax_backward_h_small_available(output)) {
                return MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H;
            }
            return MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_H;
        }
        return MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C;
    }

    if (rank - 2 == this->dim) {
        TT_ASSERT(
            this->strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H ||
                this->strategy == MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_H,
            "Invalid parallelization strategy. {} is not for dim H", this->strategy);

        if (this->strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H) {
            TT_ASSERT(
                is_moreh_softmax_backward_h_small_available(output),
                "not enough circular buffer memory for {}", this->strategy);
        }
    } else if (rank - 1 == this->dim) {
        TT_ASSERT(
            this->strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W ||
                this->strategy == MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W,
            "Invalid parallelization strategy. {} is not for dim W", this->strategy);

        if (this->strategy == MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W) {
            TT_ASSERT(
                is_moreh_softmax_backward_w_small_available(output),
                "not enough circular buffer memory for {}", this->strategy);
        }
    } else {
        TT_ASSERT(
            this->strategy == MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C,
            "Invalid parallelization strategy. large c is for dim 0 - (rank - 3)");
    }

    return this->strategy;
}

Tensor moreh_softmax_backward(
    const Tensor& output_tensor,
    const Tensor& output_grad_tensor,
    uint32_t dim,
    std::optional<Tensor> input_grad_tensor,
    const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto device = output_grad_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({output_tensor, output_grad_tensor}))};

    operation::launch_op(
        [dim, all_cores, strategy, output_mem_config, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehSoftmaxBackward{
                    .dim = dim,
                    .core_range = all_cores,
                    .op = MorehSoftmaxBackwardOp::SOFTMAX,
                    .strategy = strategy,
                    .output_mem_config = output_mem_config,
                    .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {output_tensor, output_grad_tensor},
        output_tensors,
        {},
        {input_grad_tensor});

    return output_tensors.at(0);
}

Tensor moreh_softmin_backward(
    const Tensor& output_tensor,
    const Tensor& output_grad_tensor,
    uint32_t dim,
    std::optional<Tensor> input_grad_tensor,
    const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto device = output_grad_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({output_tensor, output_grad_tensor}))};

    operation::launch_op(
        [dim, all_cores, strategy, output_mem_config, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehSoftmaxBackward{
                    .dim = dim,
                    .core_range = all_cores,
                    .op = MorehSoftmaxBackwardOp::SOFTMIN,
                    .strategy = strategy,
                    .output_mem_config = output_mem_config,
                    .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {output_tensor, output_grad_tensor},
        output_tensors,
        {},
        {input_grad_tensor});

    return output_tensors.at(0);
}

Tensor moreh_logsoftmax_backward(
    const Tensor& output_tensor,
    const Tensor& output_grad_tensor,
    uint32_t dim,
    std::optional<Tensor> input_grad_tensor,
    const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto device = output_grad_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({output_tensor, output_grad_tensor}))};

    operation::launch_op(
        [dim, all_cores, strategy, output_mem_config, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehSoftmaxBackward{
                    .dim = dim,
                    .core_range = all_cores,
                    .op = MorehSoftmaxBackwardOp::LOGSOFTMAX,
                    .strategy = strategy,
                    .output_mem_config = output_mem_config,
                    .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {output_tensor, output_grad_tensor},
        output_tensors,
        {},
        {input_grad_tensor});

    return output_tensors.at(0);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
