// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

#include <limits>
#include <optional>
#include <string>

#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"

using namespace tt::constants;

namespace reduce_op_utils {

std::map<std::string, std::string> get_defines(
    tt::tt_metal::ReduceOpMath reduce_op, tt::tt_metal::ReduceOpDim reduce_dim) {
    std::map<std::string, std::string> defines;
    // TOOD(AP): need a sync with Reduce::Max from HLK headers
    bool do_max = reduce_op == tt::tt_metal::ReduceOpMath::MAX;
    std::string reduce_dim_str;
    switch (reduce_dim) {
        case tt::tt_metal::ReduceOpDim::W: reduce_dim_str = "ReduceDim::REDUCE_ROW"; break;
        case tt::tt_metal::ReduceOpDim::H: reduce_dim_str = "ReduceDim::REDUCE_COL"; break;
        case tt::tt_metal::ReduceOpDim::HW: reduce_dim_str = "ReduceDim::REDUCE_SCALAR"; break;
        default: TT_THROW("Invalid reduce_op!");
    }
    defines["REDUCE_OP"] = (do_max ? "PoolType::MAX" : "PoolType::SUM");
    defines["REDUCE_DIM"] = reduce_dim_str;
    if (reduce_dim == tt::tt_metal::ReduceOpDim::W && reduce_op == tt::tt_metal::ReduceOpMath::SUM) {
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
    TT_FATAL((input_tensor.layout() == Layout::TILE), "Inputs to reduce must be tilized");
}

std::vector<ttnn::TensorSpec> Reduce::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    // TODO: Remove usage of input/output padded shape
    // - Get output alignment from input alignment and output dtype, layout, mem_config
    // - Get shard spec from output strides (logical shape + alignment)?
    auto output_shape = input_tensor.logical_shape();
    switch (this->dim) {
        case ReduceOpDim::H: output_shape[2] = 1; break;
        case ReduceOpDim::W: output_shape[3] = 1; break;
        case ReduceOpDim::HW:
            output_shape[2] = 1;
            output_shape[3] = 1;
            break;
    }

    TensorSpec tensor_spec(
        output_shape,
        TensorLayout(output_dtype, PageConfig(Layout::TILE), MemoryConfig(output_mem_config.buffer_type())));

    if (input_tensor.nd_shard_spec().has_value()) {
        if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            const auto& nd_shard_spec = *input_tensor.nd_shard_spec();
            return {tensor_spec.width_sharded(nd_shard_spec.grid, nd_shard_spec.orientation)};
        }

        auto nd_shard_spec = *input_tensor.nd_shard_spec();
        if (dim == ReduceOpDim::W || dim == ReduceOpDim::HW) {
            nd_shard_spec.shard_shape[-1] = 1;
        }
        if ((dim == ReduceOpDim::H || dim == ReduceOpDim::HW) && nd_shard_spec.shard_shape.rank() > 1) {
            nd_shard_spec.shard_shape[-2] = div_up(nd_shard_spec.shard_shape[-2], input_tensor.logical_shape()[-2]);
        }
        return {tensor_spec.sharded(std::move(nd_shard_spec), TensorSpec::ShardShapeAlignment::REQUIRED)};
    }

    return {tensor_spec};
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

    uint32_t num_tiles = input_tensor.physical_volume() / TILE_HW;
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
    const MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt) {
    Tensor input = input_tensor;
    if (input.layout() == Layout::ROW_MAJOR && input.storage_type() == StorageType::DEVICE) {
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

    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(ttnn::init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        std::nullopt,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true));

    if (is_multicore_hw) {
        IDevice* device;
        // Get the device
        if (input_tensor.storage_type() != StorageType::DEVICE) {
            device = ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice();
            TT_FATAL(device != nullptr, "Default device must be set if no inputs to op are on device");
        } else {
            device = input_tensor.device();
        }
        auto input_tensor_pad_shape =
            ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_tensor.padded_shape());
        auto formatted_input_tensor = input_tensor;
        if (!ttnn::operations::experimental::auto_format::AutoFormat::check_input_tensor_format(
                input_tensor, input_tensor_pad_shape)) {
            formatted_input_tensor = ttnn::operations::experimental::auto_format::AutoFormat::format_input_tensor(
                input_tensor, device, input_tensor_pad_shape, pad_value, Layout::TILE);
        }
        const Tensor output_tensor = operation::run(
                                         Reduce{
                                             reduce_math,
                                             ReduceOpDim::W,
                                             1.0,
                                             output_mem_config,
                                             output_dtype.value_or(input_tensor.dtype()),
                                             config},
                                         {formatted_input_tensor})
                                         .at(0);
        return operation::run(
                   Reduce{
                       reduce_math,
                       ReduceOpDim::H,
                       scaler,
                       output_mem_config,
                       output_dtype.value_or(input_tensor.dtype()),
                       config},
                   {output_tensor})
            .at(0);
    } else {
        return operation::run_with_autoformat(
                   Reduce{
                       reduce_math,
                       reduce_dim,
                       scaler,
                       output_mem_config,
                       output_dtype.value_or(input_tensor.dtype()),
                       config},
                   {input_tensor},
                   {},
                   {},
                   pad_value)
            .at(0);
    }
}

}  // namespace tt_metal

}  // namespace tt
