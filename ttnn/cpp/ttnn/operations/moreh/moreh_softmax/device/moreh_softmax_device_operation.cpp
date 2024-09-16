// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_device_operation.hpp"
#include <iostream>
#include "ttnn/tensor/tensor.hpp"


namespace ttnn::operations::moreh::moreh_softmax {

#define L1_512KB (512 * 1024)

bool is_moreh_softmax_w_small_available(const Tensor &tensor, const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto w = tensor.get_legacy_shape()[-1];
    int32_t Wt = (w + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;

    auto arch = tensor.device()->arch();
    const DeviceComputeKernelConfig compute_kernel_config_ =
        init_device_compute_kernel_config(tensor.device()->arch(), compute_kernel_config);
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(arch, compute_kernel_config_);

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.get_dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    auto tile_size = tt::tt_metal::detail::TileSize(data_format);
    auto intermed_tile_size = tt::tt_metal::detail::TileSize(intermed_data_format);

    int32_t cb_usage = 0;        // bytes
    cb_usage += Wt * tile_size;   // input;
    cb_usage += 1 * tile_size;   // mask;
    cb_usage += 1 * tile_size;   // scaler;

    cb_usage += Wt * tile_size;   // output;

    cb_usage += Wt * intermed_tile_size;  // exp(x);
    cb_usage += 1 * intermed_tile_size;   // reduce;
    cb_usage += 1 * intermed_tile_size;   // max;
    cb_usage += Wt * intermed_tile_size;   // x - max;
    cb_usage += 1 * intermed_tile_size;   // tmp;

    return (L1_UNRESERVED_BASE + cb_usage <= L1_512KB);
}

bool is_moreh_softmax_h_small_available(const Tensor &tensor, const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto h = tensor.get_legacy_shape()[-2];
    int32_t Ht = (h + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;

    auto arch = tensor.device()->arch();
    const DeviceComputeKernelConfig compute_kernel_config_ =
        init_device_compute_kernel_config(tensor.device()->arch(), compute_kernel_config);
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(arch, compute_kernel_config_);

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.get_dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    auto tile_size = tt::tt_metal::detail::TileSize(data_format);
    auto intermed_tile_size = tt::tt_metal::detail::TileSize(intermed_data_format);

    int32_t cb_usage = 0;        // bytes
    cb_usage += Ht * tile_size;   // input;
    cb_usage += 1 * tile_size;   // mask;
    cb_usage += 1 * tile_size;   // scaler;

    cb_usage += Ht * tile_size;   // output;

    cb_usage += Ht * intermed_tile_size;  // exp(x);
    cb_usage += 1 * intermed_tile_size;   // reduce;
    cb_usage += 1 * intermed_tile_size;   // max;
    cb_usage += Ht * intermed_tile_size;   // x - max;
    cb_usage += 1 * intermed_tile_size;   // tmp;

    return (L1_UNRESERVED_BASE + cb_usage <= L1_512KB);
}

MorehSoftmaxOperation::program_factory_t MorehSoftmaxOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (operation_attributes.strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_W) {
        return MorehSoftmaxWSmallFactory{};
    } else if(operation_attributes.strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_H) {
        return MorehSoftmaxHSmallFactory{};
    } else if(operation_attributes.strategy == MorehSoftmaxOpParallelizationStrategy::LARGE_W) {
        return MorehSoftmaxWLargeFactory{};
    } else if(operation_attributes.strategy == MorehSoftmaxOpParallelizationStrategy::LARGE_C) {
        return MorehSoftmaxCLargeFactory{};
    } else {
        return MorehSoftmaxHLargeFactory{};
    }
}

void MorehSoftmaxOperation::validate_with_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& input_tensor = tensor_args.input_tensor;
    const std::optional<Tensor>& output_tensors = tensor_args.output_tensor;
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor.get_layout() == Layout::TILE), "Inputs to softmax must be tilized");
    TT_ASSERT(input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B);
    auto rank = input_tensor.get_legacy_shape().rank();

    TT_ASSERT(
        operation_attributes.dim >= 0 && operation_attributes.dim < rank,
        "dim {} should be less than output tensor rank {}", operation_attributes.dim, rank);

    if (!output_tensors.has_value()) {
        // If the user decided to not use any optional output tensors, then this would be empty or would be a nullptr.
        return;
    }
}

void MorehSoftmaxOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_with_output_tensors(operation_attributes, tensor_args);
}

void MorehSoftmaxOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_with_output_tensors(operation_attributes, tensor_args);
}

MorehSoftmaxOperation::shape_return_value_t MorehSoftmaxOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input_tensor.get_shape();
}


MorehSoftmaxOperation::tensor_return_value_t MorehSoftmaxOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const std::optional<Tensor>& output_tensors = tensor_args.output_tensor;
    if (output_tensors.has_value()) {
        return output_tensors.value();
    }
    const auto& output_shape = input_tensor.get_legacy_shape();
    return create_device_tensor(
        output_shape,
        input_tensor.tensor_attributes->dtype,
        input_tensor.tensor_attributes->layout,
        input_tensor.device(),
        operation_attributes.output_memory_config);
}

std::tuple<MorehSoftmaxOperation::operation_attributes_t, MorehSoftmaxOperation::tensor_args_t>
MorehSoftmaxOperation::invoke(
    const Tensor &input_tensor,
    const uint32_t dim,
    const std::optional<Tensor> &output_tensor,
    const MorehSoftmaxOp op,
    const MorehSoftmaxOpParallelizationStrategy strategy,
    const std::optional<MemoryConfig> output_memory_config,
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto parallelization_strategy = MorehSoftmaxOperation::get_parallelization_strategy(
        input_tensor, output_tensor, dim, strategy, output_memory_config, compute_kernel_config);
    return {
        operation_attributes_t{
            dim, op, parallelization_strategy, output_memory_config.value_or(input_tensor.memory_config()), compute_kernel_config
        },
        tensor_args_t{
            input_tensor, output_tensor
        }
    };
}

MorehSoftmaxOpParallelizationStrategy MorehSoftmaxOperation::get_parallelization_strategy(
    const Tensor &input_tensor,
    const std::optional<Tensor> &output_tensors,
    const uint32_t dim,
    const MorehSoftmaxOpParallelizationStrategy strategy,
    const std::optional<MemoryConfig> output_memory_config,
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    const auto& input = input_tensor;

    auto rank = input.get_legacy_shape().rank();
    if (strategy == MorehSoftmaxOpParallelizationStrategy::NONE) {
        if (rank - 1 == dim) {
            if (is_moreh_softmax_w_small_available(input, compute_kernel_config)) {
                return MorehSoftmaxOpParallelizationStrategy::SMALL_W;
            }
            return MorehSoftmaxOpParallelizationStrategy::LARGE_W;
        }
        if (rank - 2 == dim) {
            if (is_moreh_softmax_h_small_available(input, compute_kernel_config)) {
                return MorehSoftmaxOpParallelizationStrategy::SMALL_H;
            }
            return MorehSoftmaxOpParallelizationStrategy::LARGE_H;
        }
        return MorehSoftmaxOpParallelizationStrategy::LARGE_C;
    }

    if (rank - 2 == dim) {
        TT_ASSERT(
            strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_H ||
                strategy == MorehSoftmaxOpParallelizationStrategy::LARGE_H,
            "Invalid parallelization strategy. {} is not for dim H", strategy);

        if (strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_H) {
            TT_ASSERT(
                is_moreh_softmax_h_small_available(input, compute_kernel_config),
                "not enough circular buffer memory for {}", strategy);
        }
    } else if (rank - 1 == dim) {
        TT_ASSERT(
            strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_W ||
                strategy == MorehSoftmaxOpParallelizationStrategy::LARGE_W,
            "Invalid parallelization strategy. {} is not for dim W", strategy);

        if (strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_W) {
            TT_ASSERT(
                is_moreh_softmax_w_small_available(input, compute_kernel_config),
                "not enough circular buffer memory for {}", strategy);
        }
    } else {
        TT_ASSERT(
            strategy == MorehSoftmaxOpParallelizationStrategy::LARGE_C,
            "Invalid parallelization strategy. large c is for dim 0 to (rank - 3)");
    }

    return strategy;
}

}
