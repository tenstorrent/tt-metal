// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

#include <limits>
#include <optional>

#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary_backward/device/unary_backward_op.hpp"

using namespace tt::constants;

namespace reduce_op_utils {

using namespace tt::tt_metal;

std::map<string, string> get_defines(ReduceOpMath reduce_op, ReduceOpDim reduce_dim) {
    std::map<string, string> defines;
    // TOOD(AP): need a sync with Reduce::Max from HLK headers
    bool do_max = reduce_op == ReduceOpMath::MAX;
    string reduce_dim_str;
    switch (reduce_dim) {
        case ReduceOpDim::W: reduce_dim_str = "ReduceDim::REDUCE_ROW"; break;
        case ReduceOpDim::H: reduce_dim_str = "ReduceDim::REDUCE_COL"; break;
        case ReduceOpDim::HW: reduce_dim_str = "ReduceDim::REDUCE_SCALAR"; break;
        default: TT_ASSERT(false && "Invalid reduce_op!");
    }
    defines["REDUCE_OP"] = (do_max ? "PoolType::MAX" : "PoolType::SUM");
    defines["REDUCE_DIM"] = reduce_dim_str;
    if (reduce_dim == ReduceOpDim::W && reduce_op == ReduceOpMath::SUM) {
        defines["REDUCE_ROW_SUM_VIA_MM"] = 1;
    }
    return defines;
}

}  // namespace reduce_op_utils
namespace tt {
namespace tt_metal {

void Reduce::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to reduce need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to reduce need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.get_layout() == Layout::TILE), "Inputs to reduce must be tilized");
    if (this->dim == ReduceOpDim::H) {
        if (input_tensor.memory_config().is_sharded()) {
            TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED, "Error");
        } else {
            TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
        }
        TT_FATAL(input_tensor.memory_config().memory_layout == this->output_mem_config.memory_layout, "Error");
    } else {
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
    }
}

std::vector<Shape> Reduce::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    auto output_shape = input_tensor.get_legacy_shape();
    auto padding = output_shape.padding();
    switch (this->dim) {
        case ReduceOpDim::H:
            output_shape[2] = TILE_HEIGHT;
            padding[2] = Padding::PadDimension{0, 31};
            break;
        case ReduceOpDim::W:
            output_shape[3] = TILE_WIDTH;
            padding[3] = Padding::PadDimension{0, 31};
            break;
        case ReduceOpDim::HW:
            output_shape[2] = TILE_HEIGHT;
            output_shape[3] = TILE_WIDTH;
            padding[2] = Padding::PadDimension{0, 31};
            padding[3] = Padding::PadDimension{0, 31};
            break;
    }
    return {Shape(output_shape, padding)};
}

std::vector<Tensor> Reduce::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        auto output_shape = this->compute_output_shapes(input_tensors).at(0);
        auto shard_spec = input_tensor.shard_spec().value();
        shard_spec.shape[0] = tt_metal::compute_volume(output_shape) / output_shape[-1];
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = shard_spec;
        return {
            create_device_tensor(output_shape, this->output_dtype, Layout::TILE, input_tensor.device(), mem_config)};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Reduce::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy) {
        case ReduceOpParallelizationStrategy::MULTI_CORE_H:
            return reduce_multi_core_h(input_tensor, output_tensor, this->math_op, compute_kernel_config, this->scaler);
        case ReduceOpParallelizationStrategy::MULTI_CORE_W:
            return reduce_multi_core_w(input_tensor, output_tensor, this->math_op, compute_kernel_config, this->scaler);
        case ReduceOpParallelizationStrategy::MULTI_CORE_HW:
        case ReduceOpParallelizationStrategy::SINGLE_CORE_HW:
            return reduce_single_core_hw(
                input_tensor, output_tensor, this->math_op, compute_kernel_config, this->scaler);
        default: TT_THROW("Unsupported parallelization strategy");
    }
}

ReduceOpParallelizationStrategy Reduce::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    uint32_t num_tiles = input_tensor.volume() / TILE_HW;
    if (this->dim == ReduceOpDim::H) {
        return ReduceOpParallelizationStrategy::MULTI_CORE_H;
    } else if (this->dim == ReduceOpDim::W) {
        return ReduceOpParallelizationStrategy::MULTI_CORE_W;
    } else if (this->dim == ReduceOpDim::HW) {
        if (num_tiles > 1) {
            return ReduceOpParallelizationStrategy::MULTI_CORE_HW;
        } else {
            return ReduceOpParallelizationStrategy::SINGLE_CORE_HW;
        }
    } else {
        TT_THROW("Unsupported reduce dim");
    }
}

// reduce min
// reduce min = - reduce_max( -x )
Tensor reduce_min(
    const Tensor& input_tensor,
    ReduceOpDim reduce_dim,
    float scaler = 1.0f,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt) {
    Tensor input = input_tensor;
    if (input.get_layout() == Layout::ROW_MAJOR && input.storage_type() == StorageType::DEVICE) {
        input = ttnn::operations::unary_backward::change_layout_to_tile(input, output_mem_config);
    }
    Tensor n_input_tensor = ttnn::neg(input, output_mem_config);
    Tensor max_reduce = reduce(
        n_input_tensor, ReduceOpMath::MAX, reduce_dim, scaler, output_mem_config, std::nullopt, compute_kernel_config);
    Tensor min_tensor = ttnn::neg(max_reduce, output_mem_config);
    return min_tensor;
}

Tensor reduce(
    const Tensor& input_tensor,
    ReduceOpMath reduce_math,
    ReduceOpDim reduce_dim,
    float scaler,
    const MemoryConfig& output_mem_config,
    const std::optional<DataType>& output_dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    if (reduce_math == ReduceOpMath::MIN) {
        return reduce_min(input_tensor, reduce_dim, scaler, output_mem_config);
    }

    auto parallelization_strategy =
        Reduce{reduce_math, reduce_dim, scaler, output_mem_config}.get_parallelization_strategy({input_tensor});
    auto is_multicore_hw = parallelization_strategy == ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    float pad_value = reduce_math == ReduceOpMath::MAX ? -std::numeric_limits<float>::infinity() : 0;

    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(
        ttnn::init_device_compute_kernel_config(input_tensor.device()->arch(), std::nullopt, MathFidelity::HiFi4));

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    if (is_multicore_hw) {
        operation::launch_op(
            [reduce_math, reduce_dim, pad_value, scaler, output_dtype, output_mem_config, config](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto& input_tensor = input_tensors.at(0);
                Device* device;

                // Get the device
                if (input_tensor.storage_type() != StorageType::DEVICE) {
                    device = ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice();
                    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
                } else {
                    device = input_tensor.device();
                }
                auto input_tensor_pad_shape = ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
                auto formatted_input_tensor = input_tensor;
                if (!ttnn::operations::experimental::auto_format::AutoFormat::check_input_tensor_format(input_tensor, input_tensor_pad_shape)) {
                    formatted_input_tensor = ttnn::operations::experimental::auto_format::AutoFormat::format_input_tensor(
                        input_tensor, device, input_tensor_pad_shape, pad_value, Layout::TILE);
                }
                const Tensor output_tensor = operation::run_without_autoformat(
                                                 Reduce{
                                                     reduce_math,
                                                     ReduceOpDim::W,
                                                     1.0,
                                                     output_mem_config,
                                                     output_dtype.value_or(input_tensor.get_dtype()),
                                                     config},
                                                 {formatted_input_tensor})
                                                 .at(0);
                return operation::run_without_autoformat(
                    Reduce{
                        reduce_math,
                        ReduceOpDim::H,
                        scaler,
                        output_mem_config,
                        output_dtype.value_or(input_tensor.get_dtype()),
                        config},
                    {output_tensor});
            },
            {input_tensor},
            output_tensors);
    } else {
        operation::launch_with_autoformat(
            [reduce_math, reduce_dim, pad_value, scaler, output_dtype, output_mem_config, config](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto& input_tensor = input_tensors.at(0);
                return operation::run_with_autoformat(
                    Reduce{
                        reduce_math,
                        reduce_dim,
                        scaler,
                        output_mem_config,
                        output_dtype.value_or(input_tensor.get_dtype()),
                        config},
                    {input_tensor},
                    {},
                    {},
                    pad_value);
            },
            {input_tensor},
            output_tensors);
    }
    return output_tensors.at(0);
}
}  // namespace tt_metal

}  // namespace tt
