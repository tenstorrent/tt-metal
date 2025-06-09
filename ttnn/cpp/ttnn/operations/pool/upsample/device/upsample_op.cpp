// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample_op.hpp"

#include "ttnn/tensor/types.hpp"
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::upsample {
using namespace tt;
using namespace tt::tt_metal;

void UpSample::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to copy need to be allocated in buffers on device!");
    // TT_FATAL(input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Input tensor layout should be ROW_MAJOR");
    TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16, "Input tensor data type should be BFLOAT16");
    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == output_mem_config_.memory_layout(),
            "Input tensor memory layout should be same as output tensor memory layout");
        if (mode_ == "nearest") {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
                    input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
                "Input tensor memory layout should be HEIGHT or BLOCK sharded");
        } else if (mode_ == "bilinear") {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                "Input tensor memory layout should be HEIGHT sharded");
        }
        TT_FATAL(mode_ == "bilinear" || mode_ == "nearest", "Upsample only supports bilinear or nearest mode");
        TT_FATAL(
            input_tensor_a.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
            "Input buffer should be sharded in L1");
    }
}

std::vector<TensorSpec> UpSample::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    // NOTE1: data is packed in { N, H , W, C }
    // NOTE2: Mapping it into in 2D format should be {N*H*W, C}
    // NOTE3: Assuming output data type is same as input
    const auto& input = input_tensors.at(0);
    const auto input_shape = input.logical_shape();

    uint32_t out_n = input_shape[0];
    uint32_t out_h = input_shape[1] * scale_factor_h_;
    uint32_t out_w = input_shape[2] * scale_factor_w_;
    uint32_t out_c = input_shape[3];

    ttnn::Shape output_shape = ttnn::Shape({out_n, out_h, out_w, out_c});

    if (output_mem_config_.is_sharded()) {
        TT_FATAL(
            input.memory_config().is_sharded(),
            "Output memory config is sharded but input memory config is not sharded");
        TT_FATAL(
            input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Input memory config is not HEIGHT or BLOCK sharded");
        TT_FATAL(
            input.memory_config().shard_spec()->grid.ranges().size() == 1 ||
                input.memory_config().memory_layout() != TensorMemoryLayout::BLOCK_SHARDED,
            "Block sharded input should have only one CoreRange");

        auto shard_spec = output_mem_config_.shard_spec().value();
        shard_spec.shape = {
            input.shard_spec()->shape[0] * scale_factor_h_ * scale_factor_w_, input.shard_spec()->shape[1]};
        MemoryConfig mem_config = output_mem_config_.with_shard_spec(shard_spec);
        return {TensorSpec(output_shape, TensorLayout(input.dtype(), PageConfig(input.layout()), mem_config))};
    }

    return {TensorSpec(output_shape, TensorLayout(input.dtype(), PageConfig(input.layout()), output_mem_config_))};
}

operation::ProgramWithCallbacks UpSample::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const Tensor& input_tensor_0 = input_tensors.at(0);
    Tensor& output_tensor_0 = output_tensors.at(0);
    switch (get_parallelization_strategy(input_tensors)) {
        case UpSampleParallelizationStrategy::MULTI_CORE:
            if (mode_ == "bilinear") {
                return bilinear_multi_core(
                    input_tensor_0, output_tensor_0, scale_factor_h_, scale_factor_w_, this->compute_kernel_config_);
            } else if (mode_ == "nearest") {
                return upsample_multi_core(input_tensor_0, output_tensor_0, scale_factor_h_, scale_factor_w_);
            } else {
                TT_THROW("Unsupported mode");
            }
        case UpSampleParallelizationStrategy::SINGLE_CORE:
            if (mode_ == "nearest") {
                return upsample_single_core(input_tensor_0, output_tensor_0, scale_factor_h_, scale_factor_w_);
            } else {
                TT_THROW("Unsupported mode");
            }
    };
    return upsample_single_core(input_tensor_0, output_tensor_0, scale_factor_h_, scale_factor_w_);
}

UpSampleParallelizationStrategy UpSample::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    auto input = input_tensors.at(0);
    if (input.memory_config().is_sharded()) {
        return UpSampleParallelizationStrategy::MULTI_CORE;
    }
    return UpSampleParallelizationStrategy::SINGLE_CORE;
}

}  // namespace ttnn::operations::upsample
