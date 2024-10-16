// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample_op.hpp"

#include <algorithm>
#include <cmath>

#include "detail/util.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/host_api.hpp"

namespace ttnn::operations::upsample {
using namespace tt;

void UpSample::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to copy need to be allocated in buffers on device!");
    // TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Input tensor layout should be ROW_MAJOR");
    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16, "Input tensor data type should be BFLOAT16");
    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == output_mem_config_.memory_layout,
                 "Input tensor memory layout should be same as output tensor memory layout");
        if (mode_ == "nearest")
            TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
                         input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED,
                     "Input tensor memory layout should be HEIGHT or BLOCK sharded");
        else if (mode_ == "bilinear")
            TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
                     "Input tensor memory layout should be HEIGHT sharded");
        TT_FATAL(mode_ == "bilinear" || mode_ == "nearest", "Upsample only supports bilinear or nearest mode");
        TT_FATAL(input_tensor_a.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
                 "Input buffer should be sharded in L1");
    }
}

std::vector<tt::tt_metal::LegacyShape> UpSample::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // NOTE1: data is packed in { N, H , W, C }
    // NOTE2: Mapping it into in 2D format should be {N*H*W, C}
    // NOTE3: Assuming output data type is same as input
    const auto& input = input_tensors.at(0);
    const auto input_shape = input.get_logical_shape();

    uint32_t out_n = input_shape[0];
    uint32_t out_h = input_shape[1] * scale_factor_h_;
    uint32_t out_w = input_shape[2] * scale_factor_w_;
    uint32_t out_c = input_shape[3];
    const auto out_dims = std::vector<uint32_t>({out_n, out_h, out_w, out_c});  // in the NHWC format

    return {tt::tt_metal::LegacyShape{out_dims}};
}

std::vector<Tensor> UpSample::create_output_tensors(const std::vector<Tensor>& inputs) const {
    const auto& input = inputs.at(0);
    if (output_mem_config_.is_sharded()) {
        if (input.memory_config().is_sharded()) {
            auto mem_config = output_mem_config_;
            auto input_shard_spec = input.memory_config().shard_spec.value();
            auto output_shape = compute_output_shapes(inputs).at(0);
            if (input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                auto ncores = input_shard_spec.num_cores();
                array<uint32_t, 2> output_shard_shape = {
                    div_up(output_shape[0] * output_shape[1] * output_shape[2], ncores), output_shape[-1]};
                auto output_shard_spec = input_shard_spec;
                output_shard_spec.shape = output_shard_shape;
                mem_config.shard_spec = output_shard_spec;
                log_debug(LogOp, "output_shard_shape: {}", output_shard_shape);
                log_debug(LogOp, "output_shard_spec: {}", output_shard_spec);
                return {create_device_tensor(
                    output_shape, input.get_dtype(), input.get_layout(), input.device(), mem_config)};
            } else if (input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                auto shard_grid = input_shard_spec.grid.ranges();
                TT_FATAL(shard_grid.size() == 1, "Block sharded input should have only one CoreRange");
                auto core_range = *shard_grid.begin();
                uint32_t ncores_w = core_range.end_coord.x + 1;
                uint32_t ncores_h = core_range.end_coord.y + 1;
                // array<uint32_t, 2> output_shard_shape = {output_shape[0] * output_shape[1] * output_shape[2] /
                // ncores_h, output_shape[-1] / ncores_w}; auto output_shard_spec = input_shard_spec;
                // output_shard_spec.shape = output_shard_shape;
                // mem_config.shard_spec = output_shard_spec;
                auto output_shard_spec = mem_config.shard_spec.value();
                auto output_shard_shape = output_shard_spec.shape;
                log_debug(LogOp, "ncores_w, ncores_h: {} {}", ncores_w, ncores_h);
                log_debug(LogOp, "output_shard_shape: {}", output_shard_shape);
                return {create_device_tensor(
                    output_shape, input.get_dtype(), input.get_layout(), input.device(), mem_config)};
            } else {
                TT_THROW("input memory config is not HEIGHT or BLOCK sharded");
            }
        } else {
            TT_THROW("Output memory config is sharded but input memory config is not sharded");
        }
    } else {
        return operation::generic_create_output_tensors(
            *this, inputs, input.get_dtype(), input.get_layout(), output_mem_config_);
    }
}

operation::ProgramWithCallbacks UpSample::create_program(const std::vector<Tensor>& input_tensors,
                                                         std::vector<Tensor>& output_tensors) const {
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
            if (mode_ == "nearest")
                return upsample_single_core(input_tensor_0, output_tensor_0, scale_factor_h_, scale_factor_w_);
            else {
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
