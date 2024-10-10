// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_softmax {

#define L1_512KB (512 * 1024)

bool is_moreh_softmax_w_small_available(const Tensor& tensor, const DeviceComputeKernelConfig& compute_kernel_config) {
    auto w = tensor.get_shape()[-1];
    int32_t Wt = (w + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;

    auto arch = tensor.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.get_dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    auto tile_size = tt::tt_metal::detail::TileSize(data_format);
    auto intermed_tile_size = tt::tt_metal::detail::TileSize(intermed_data_format);

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

    return (tensor.device()->get_base_allocator_addr(HalMemType::L1) + cb_usage <= L1_512KB);
}

bool is_moreh_softmax_h_small_available(const Tensor& tensor, const DeviceComputeKernelConfig& compute_kernel_config) {
    auto h = tensor.get_shape()[-2];
    int32_t Ht = (h + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;

    auto arch = tensor.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.get_dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    auto tile_size = tt::tt_metal::detail::TileSize(data_format);
    auto intermed_tile_size = tt::tt_metal::detail::TileSize(intermed_data_format);

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

    return (tensor.device()->get_base_allocator_addr(HalMemType::L1) + cb_usage <= L1_512KB);
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
    TT_FATAL((input.get_layout() == Layout::TILE), "Inputs to softmax must be tilized");
    TT_FATAL(
        input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::BFLOAT8_B,
        "Inputs must be of bfloat16 or bfloat8_b type");

    const auto rank = input.get_shape().rank();
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

MorehSoftmaxOperation::shape_return_value_t MorehSoftmaxOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input.get_shape();
}

MorehSoftmaxOperation::tensor_return_value_t MorehSoftmaxOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output = tensor_args.output;
    if (output.has_value())
        return output.value();

    const auto& input = tensor_args.input;
    const auto& output_shape = input.get_shape();
    return create_device_tensor(
        output_shape, input.get_dtype(), input.get_layout(), input.device(), operation_attributes.memory_config);
}

std::tuple<MorehSoftmaxOperation::operation_attributes_t, MorehSoftmaxOperation::tensor_args_t>
MorehSoftmaxOperation::invoke(
    const Tensor& input,
    uint32_t dim,
    const std::optional<Tensor>& output,
    const MorehSoftmaxOp op,
    const MorehSoftmaxOpParallelizationStrategy strategy,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            dim,
            op,
            strategy,
            memory_config.value_or(input.memory_config()),
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
        tensor_args_t{input, output}};
}

MorehSoftmaxOpParallelizationStrategy MorehSoftmaxOperation::get_parallelization_strategy(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto strategy = operation_attributes.strategy;
    const auto dim = operation_attributes.dim;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto rank = input.get_shape().rank();
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
