// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_kv_cache_load_slice_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"

namespace ttnn::operations::experimental::transformer {

// NLP KV Cache Unpad To Sharded op
void NlpKVCacheLoadSliceDeviceOperation::validate(const std::vector<Tensor> &input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "Error");

    for (uint32_t i = 0; i < input_tensor_a.get_shape().with_tile_padding().rank(); i++) {
        TT_FATAL(this->output_tensor_start[i] < input_tensor_a.get_shape().with_tile_padding()[i], "Error");
        TT_FATAL(this->output_tensor_end[i] < input_tensor_a.get_shape().with_tile_padding()[i], "Error");

        // Check if start shape is <= end shape
        TT_FATAL(this->output_tensor_start[i] <= this->output_tensor_end[i], "Error");
    }

    ttnn::Shape output_tensor_shape = this->compute_output_shapes(input_tensors)[0];
    auto num_dims = input_tensor_a.get_shape().with_tile_padding().rank();
    TT_FATAL(num_dims == 4, "Input tensor must be 4D");
    const auto input_shape = input_tensor_a.get_shape().with_tile_padding();
    auto dim0 = input_shape[0];
    auto dim1 = input_shape[1];
    auto fused_batch_heads = dim0 * dim1;
    auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
    // Need at least fused_batch_heads cores to unpad into sharded tensor
    TT_FATAL(fused_batch_heads <= core_grid.x * core_grid.y, "Error");
    TT_FATAL(input_tensor_a.volume() % TILE_HW == 0, "Error");
    TT_FATAL((output_tensor_shape[-2] % TILE_HEIGHT == 0) &&
                (this->output_tensor_start[-2] % TILE_HEIGHT == 0),
            "Can only unpad tilized tensor with full tiles");
    TT_FATAL((output_tensor_shape[-1] % TILE_WIDTH == 0) &&
                (this->output_tensor_start[-1] % TILE_WIDTH == 0),
            "Can only unpad tilized tensor with full tiles");
}
std::vector<ttnn::Shape> NlpKVCacheLoadSliceDeviceOperation::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    std::vector<uint32_t> out_shape;
    auto rank = input_tensors[0].get_shape().with_tile_padding().rank();
    out_shape.reserve(rank);
    for (uint32_t i = 0; i < rank; i++) {
        out_shape.push_back(this->output_tensor_end[i] - this->output_tensor_start[i] + 1);
    }
    ttnn::Shape output_tensor_shape(out_shape);
    return {output_tensor_shape};
}
std::vector<Tensor> NlpKVCacheLoadSliceDeviceOperation::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto input_shape = input_tensor_a.get_shape().with_tile_padding();
    auto dim0 = input_shape[0];
    auto dim1 = input_shape[1];
    auto unpad_length = this->output_tensor_end[2] - this->output_tensor_start[2] + 1;
    auto head_dim = input_shape[3];
    auto fused_batch_heads = dim0 * dim1;

    auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
    auto shard_grid = num_cores_to_corerange_set(fused_batch_heads, core_grid, true);
    ShardSpec shard_spec{shard_grid, {unpad_length, head_dim}};
    auto mem_config = tt::tt_metal::MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1};
    mem_config.shard_spec = shard_spec;

    return {create_device_tensor(
        this->compute_output_shapes(input_tensors).at(0),
        input_tensor_a.get_dtype(),
        input_tensor_a.get_layout(),
        input_tensor_a.device(),
        mem_config
        )};
}
operation::ProgramWithCallbacks NlpKVCacheLoadSliceDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return multi_core_nlp_kv_cache_load_slice(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end);
}


}  // namespace ttnn::operations::experimental::transformer
