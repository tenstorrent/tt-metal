// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"


namespace ttnn::operations::kv_cache {

using namespace tt::constants;

void UpdateCache::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& cache_tensor = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(1);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE and cache_tensor.storage_type() == StorageType::DEVICE, "Operands to update_cache need to be on device!");
    TT_FATAL(input_tensor.device() == cache_tensor.device(), "Operands to update_cache need to be on the same device!");
    TT_FATAL(input_tensor.buffer() != nullptr and cache_tensor.buffer() != nullptr, "Operands to update_cache need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.get_layout() == Layout::TILE && cache_tensor.get_layout() == Layout::TILE), "Inputs to update_cache must be tilized");
    TT_FATAL(input_tensor.get_dtype() == DataType::FLOAT32 || input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B, "Error");
    TT_FATAL(cache_tensor.get_dtype() == DataType::FLOAT32 || cache_tensor.get_dtype() == DataType::BFLOAT16 || cache_tensor.get_dtype() == DataType::BFLOAT8_B, "Error");

    TT_FATAL(input_tensor.get_shape().with_tile_padding()[-1] == cache_tensor.get_shape().with_tile_padding()[-1], "Error");
    TT_FATAL(input_tensor.get_shape().with_tile_padding()[0] == 1, "Error");
    TT_FATAL(input_tensor.get_shape().with_tile_padding()[1] == cache_tensor.get_shape().with_tile_padding()[1], "Error");
    TT_FATAL(cache_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
    if (this->op_type == UpdateCacheOpType::FILL) {
        // TODO: If we want to support mixed precision like decode, we need to add simple compute kernel for conversion
        TT_FATAL(input_tensor.get_dtype() == cache_tensor.get_dtype(), "Input and cache tensors must have same dtype!");

        // TODO: For interleaved, assume each core handles 1 tile of seq_len if kv_heads > 1
        // For 56 cores and 2 heads, this effectively caps max seq len at 56 / 2 * 32 = 896
        // Can generalize interleaved to infer and check arbitrary number of tiles along seq_len per core; or, add more robust logic in reader/writer loops to handle generic blocking of work
        // For sharded, we infer number of tiles each core handles from shard so no issues there
        if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED and input_tensor.get_shape().with_tile_padding()[1] > 1) {
            const uint32_t num_blocks_of_work = input_tensor.get_shape().with_tile_padding()[1] * input_tensor.get_shape().with_tile_padding()[-2] / TILE_HEIGHT;
            const auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
            TT_FATAL((num_blocks_of_work <= compute_with_storage_grid_size.x * compute_with_storage_grid_size.y), "Error");
        }

        if (input_tensor.is_sharded()) {
            TT_FATAL(input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED, "Error");
            TT_FATAL(input_tensor.shard_spec().value().shape[1] == input_tensor.get_shape().with_tile_padding()[-1], "Error");
            // Require even work division along seq_len and also only 1 head per core
            TT_FATAL(input_tensor.get_shape().with_tile_padding()[-2] % input_tensor.shard_spec().value().shape[0] == 0, "Seq len must be divisible by shard height!");
        }

        TT_FATAL(this->batch_idx < cache_tensor.get_shape().with_tile_padding()[0], "Error");
        TT_FATAL(input_tensor.get_shape().with_tile_padding()[-2] <= cache_tensor.get_shape().with_tile_padding()[-2], "Error");
    } else if (this->op_type == UpdateCacheOpType::UPDATE) {
        if (input_tensor.device()->arch() == tt::ARCH::GRAYSKULL) {
            TT_FATAL(cache_tensor.get_dtype() == DataType::BFLOAT16, "#12931: Update Cache currently produces non-deterministic output on GS when converting data types for cache tensor");
        }
        if (input_tensor.is_sharded()) {
            TT_FATAL(input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED, "Error");
            TT_FATAL(input_tensor.shard_spec().value().shape[1] == input_tensor.get_shape().with_tile_padding()[-1], "Error");
            // Require even work division for now
            TT_FATAL((input_tensor.volume() / input_tensor.get_shape().with_tile_padding()[-1]) % input_tensor.shard_spec().value().shape[0] == 0, "Error");
        } else {
            TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
        }
        TT_FATAL(cache_tensor.get_shape().with_tile_padding()[0] <= input_tensor.get_shape().with_tile_padding()[-2], "Error");
        // batch offset is only valid if num_user less than 32 and batch_offset + num_user <= 32
        if (cache_tensor.get_shape().with_tile_padding()[0] < 32) TT_FATAL(this->batch_offset + cache_tensor.get_shape().with_tile_padding()[0] <= 32, "Error");
        else TT_FATAL(this->batch_offset == 0, "Error");
    }
}

std::vector<ttnn::Shape> UpdateCache::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

std::vector<Tensor> UpdateCache::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

operation::ProgramWithCallbacks UpdateCache::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& cache_tensor = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(1);

    switch(this->get_parallelization_strategy(input_tensors)) {
        case UpdateCacheOpParallelizationStrategy::MULTI_CORE:
        default:
            if (this->op_type == UpdateCacheOpType::FILL) {
                return fill_cache_multi_core(cache_tensor, input_tensor, this->batch_idx, this->update_idx);
            } else {
                return update_cache_multi_core(cache_tensor, input_tensor, this->update_idx, this->batch_offset, this->compute_kernel_config);
            }
    };
}


UpdateCacheOpParallelizationStrategy UpdateCache::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    return UpdateCacheOpParallelizationStrategy::MULTI_CORE;
}

const operation::Hash UpdateCache::compute_program_hash(
    const std::vector<Tensor> &input_tensors) const {
    return operation::hash_operation<UpdateCache>(this->op_type, input_tensors);
}

} // ttnn::operations::kv_cache
