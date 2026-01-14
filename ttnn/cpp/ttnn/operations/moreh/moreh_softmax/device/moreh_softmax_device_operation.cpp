// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_softmax {

#define L1_512KB (512 * 1024)

bool is_moreh_softmax_w_small_available(const Tensor& tensor, const DeviceComputeKernelConfig& compute_kernel_config) {
    auto w = tensor.logical_shape()[-1];
    int32_t Wt = (w + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;

    auto arch = tensor.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    auto tile_size = tt::tile_size(data_format);
    auto intermed_tile_size = tt::tile_size(intermed_data_format);

    int32_t cb_usage = 0;        // bytes
    cb_usage += Wt * tile_size;  // input;
    cb_usage += 1 * tile_size;   // mask;
    cb_usage += 1 * tile_size;   // scaler;

    cb_usage += Wt * tile_size;  // output;

    cb_usage += Wt * intermed_tile_size;  // exp(x);
    cb_usage += 1 * intermed_tile_size;   // reduce;
    cb_usage += 1 * intermed_tile_size;   // max;
    cb_usage += Wt * intermed_tile_size;  // x - max;
    cb_usage += 1 * intermed_tile_size;   // tmp;

    return (tensor.device()->allocator()->get_base_allocator_addr(HalMemType::L1) + cb_usage <= L1_512KB);
}

bool is_moreh_softmax_h_small_available(const Tensor& tensor, const DeviceComputeKernelConfig& compute_kernel_config) {
    auto h = tensor.logical_shape()[-2];
    int32_t Ht = (h + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;

    auto arch = tensor.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    auto tile_size = tt::tile_size(data_format);
    auto intermed_tile_size = tt::tile_size(intermed_data_format);

    int32_t cb_usage = 0;        // bytes
    cb_usage += Ht * tile_size;  // input;
    cb_usage += 1 * tile_size;   // mask;
    cb_usage += 1 * tile_size;   // scaler;

    cb_usage += Ht * tile_size;  // output;

    cb_usage += Ht * intermed_tile_size;  // exp(x);
    cb_usage += 1 * intermed_tile_size;   // reduce;
    cb_usage += 1 * intermed_tile_size;   // max;
    cb_usage += Ht * intermed_tile_size;  // x - max;
    cb_usage += 1 * intermed_tile_size;   // tmp;

    return (tensor.device()->allocator()->get_base_allocator_addr(HalMemType::L1) + cb_usage <= L1_512KB);
}

MorehSoftmaxOperation::program_factory_t MorehSoftmaxOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto strategy = get_parallelization_strategy(operation_attributes, tensor_args);

    switch (strategy) {
        case MorehSoftmaxOpParallelizationStrategy::SMALL_W: return MorehSoftmaxWSmallFactory{};
        case MorehSoftmaxOpParallelizationStrategy::SMALL_H: return MorehSoftmaxHSmallFactory{};
        case MorehSoftmaxOpParallelizationStrategy::LARGE_W: return MorehSoftmaxWLargeFactory{};
        case MorehSoftmaxOpParallelizationStrategy::LARGE_C: return MorehSoftmaxCLargeFactory{};
        default: return MorehSoftmaxHLargeFactory{};
    }
}

void MorehSoftmaxOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL((input.layout() == Layout::TILE), "Inputs to softmax must be tilized");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::BFLOAT8_B ||
            input.dtype() == DataType::FLOAT32,
        "Inputs must be of bfloat16, bfloat8_b or float32 type. Received: {}",
        input.dtype());
    const auto rank = input.logical_shape().rank();
    const auto dim = operation_attributes.dim;
    TT_FATAL(dim >= 0 && dim < rank, "dim {} should be less than output tensor rank {}", dim, rank);
}

void MorehSoftmaxOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

void MorehSoftmaxOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

MorehSoftmaxOperation::spec_return_value_t MorehSoftmaxOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->tensor_spec();
    }

    return TensorSpec(
        tensor_args.input.logical_shape(),
        TensorLayout(
            tensor_args.input.dtype(), PageConfig(tensor_args.input.layout()), operation_attributes.memory_config));
}

MorehSoftmaxOperation::tensor_return_value_t MorehSoftmaxOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output = tensor_args.output;
    if (output.has_value()) {
        return output.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

MorehSoftmaxOpParallelizationStrategy MorehSoftmaxOperation::get_parallelization_strategy(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto strategy = operation_attributes.strategy;
    const auto dim = operation_attributes.dim;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto rank = input.logical_shape().rank();
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
        TT_FATAL(
            strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_H ||
                strategy == MorehSoftmaxOpParallelizationStrategy::LARGE_H,
            "Invalid parallelization strategy. {} is not for dim H",
            strategy);

        if (strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_H) {
            TT_FATAL(
                is_moreh_softmax_h_small_available(input, compute_kernel_config),
                "not enough circular buffer memory for {}",
                strategy);
        }
    } else if (rank - 1 == dim) {
        TT_FATAL(
            strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_W ||
                strategy == MorehSoftmaxOpParallelizationStrategy::LARGE_W,
            "Invalid parallelization strategy. {} is not for dim W",
            strategy);

        if (strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_W) {
            TT_FATAL(
                is_moreh_softmax_w_small_available(input, compute_kernel_config),
                "not enough circular buffer memory for {}",
                strategy);
        }
    } else {
        TT_FATAL(
            strategy == MorehSoftmaxOpParallelizationStrategy::LARGE_C,
            "Invalid parallelization strategy. large c is for dim 0 to (rank - 3)");
    }

    return strategy;
}

}  // namespace ttnn::operations::moreh::moreh_softmax

namespace ttnn::prim {
ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation::tensor_return_value_t moreh_softmax(
    const Tensor& input,
    uint32_t dim,
    const std::optional<Tensor>& output,
    const ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOp op,
    const ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy strategy,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation;
    const bool is_fp32 = input.dtype() == DataType::FLOAT32;
    auto operation_attributes = OperationType::operation_attributes_t{
        dim,
        op,
        strategy,
        memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(
            input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32)};
    auto tensor_args = OperationType::tensor_args_t{input, output};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
