// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_device_operation.hpp"
#include "rotary_embedding_program_factory.hpp"

#include "ttnn/operations/core/work_split/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void RotaryEmbedding::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& cos = input_tensors.at(1);
    const auto& sin = input_tensors.at(2);
    TT_FATAL(input_tensors.size() == 3);
    auto ref_device = input_tensor.device();
    for (const auto& input : input_tensors) {
        TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to rotary embedding need to be on device!");
        TT_FATAL(input.buffer() != nullptr, "Operands to rotary embedding need to be allocated in buffers on device!");
        TT_FATAL(input.device() == ref_device, "Operands to rotary embedding need to be on same device!");
        TT_FATAL((input.get_layout() == Layout::TILE), "Inputs to rotary embedding must be tilized");
    }

    TT_FATAL(input_tensor.get_legacy_shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
    uint32_t seq_len = input_tensor.get_legacy_shape()[-2];
    uint32_t B = input_tensor.get_legacy_shape()[0];
    uint32_t X = input_tensor.get_legacy_shape()[-1];
    TT_FATAL(cos.get_dtype() == sin.get_dtype(), "Cos and Sin dtypes must match");
    TT_FATAL(cos.get_legacy_shape() == sin.get_legacy_shape(), "Cos and Sin dims must match");
    TT_FATAL(cos.get_legacy_shape()[0] == 1 && cos.get_legacy_shape()[1] == 1 && cos.get_legacy_shape()[-1] == X, "Cos dims must match input dims");
    if (this->token_idx.has_value()) {
        TT_FATAL(cos.get_legacy_shape()[-2] >= token_idx, "Cos dims must match input dims");
    } else {
        TT_FATAL(cos.get_legacy_shape()[-2] >= seq_len, "Cos dims must match input dims");
    }
    if (input_tensor.is_sharded()) {
        TT_FATAL(input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
        TT_FATAL(input_tensor.shard_spec().value().shape[1] == input_tensor.get_legacy_shape()[-1]);
        // Require even work division for now
        TT_FATAL(
            (input_tensor.volume() / input_tensor.get_legacy_shape()[-1]) % input_tensor.shard_spec().value().shape[0] == 0);
        if (this->output_mem_config.is_sharded()) {
            TT_FATAL(this->output_mem_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
        }
    } else if (this->output_mem_config.is_sharded()) {
        TT_FATAL(this->output_mem_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
    } else {
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<Shape> RotaryEmbedding::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.get_legacy_shape();
    if (!this->token_idx.has_value()) {
        shape[-2] = round_up(this->seq_len, TILE_HEIGHT);
    }
    return {shape};
}

std::vector<Tensor> RotaryEmbedding::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto output_shape = this->compute_output_shapes(input_tensors)[0];
    if (this->output_mem_config.is_sharded()) {
        ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
        if (input_tensor.is_sharded()) {
            shard_spec = input_tensor.shard_spec().value();
        } else {
            uint32_t num_blocks = input_tensor.volume() / input_tensor.get_legacy_shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
            uint32_t num_grid_cores = core_grid.x * core_grid.y;
            uint32_t num_cores = 0;
            for (uint32_t i = num_grid_cores; i > 0; --i) {
                if (num_blocks % i == 0) {
                    num_cores = i;
                    break;
                }
            }
            uint32_t Ht = div_up(num_blocks, num_cores);
            shard_spec.grid = ttnn::operations::core::work_split::num_cores_to_corerange_set(num_cores, core_grid, true);
            shard_spec.shape = {Ht * TILE_HEIGHT, input_tensor.get_legacy_shape()[-1]};
            shard_spec.orientation = ShardOrientation::ROW_MAJOR;
        }
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = shard_spec;
        return {create_device_tensor(
            output_shape,
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            input_tensor.device(),
            mem_config)};
    } else {
        return {create_device_tensor(
            output_shape, input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), this->output_mem_config)};
    }
}

operation::ProgramWithCallbacks RotaryEmbedding::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& cos = input_tensors.at(1);
    const auto& sin = input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);

    switch (this->get_parallelization_strategy(input_tensors)) {
        case RotaryEmbeddingOpParallelizationStrategy::MULTI_CORE:
        default:
            return rotary_embedding_multi_core(input_tensor, cos, sin, output_tensor, this->token_idx, this->compute_kernel_config);
    }
}

RotaryEmbeddingOpParallelizationStrategy RotaryEmbedding::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    return RotaryEmbeddingOpParallelizationStrategy::MULTI_CORE;
}

const operation::Hash RotaryEmbedding::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    return operation::hash_operation<RotaryEmbedding>(this->seq_len, this->output_mem_config, input_tensors);
}

}  // namespace tt_metal
}  // namespace tt
